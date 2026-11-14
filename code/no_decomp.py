import numpy as np
import GPy
from utils import reset_kernel_hypers, gen_name, invgamma_solver
from priors import InverseGamma, Exponential
import string

def generate_kernel_list(M, dim, RandomState, kernel=GPy.kern.RBF, ARD=False):
    """
    Generate a kernel_list given M groups such that
    concentration parameters: alpha = ones(M)
    mixing proportions: eta ~ Dir(alpha)
    assignments: z_d ~ Cat(eta), for d in range(dim)
    """
    alpha = np.ones(M)
    z = RandomState.choice(M, size=dim, p=RandomState.dirichlet(alpha))
    return [kernel(input_dim=dim, active_dims=np.where(z==v)[0], ARD=ARD) for v in np.unique(z)]

class SplitMergeProposal():
    """
    Transition kernel for Metropolis Hastings sampler.
    """
    def __init__(self, kernel=GPy.kern.RBF, lengthscale=None, variance=None,
                 ARD=False, RandomState=None):
        self.lengthscale = lengthscale
        self.variance = variance
        self.ARD = ARD
        self.kernel = kernel
        self.RandomState = np.random if RandomState is None else RandomState


    def split_kernel_random(self, kernel_list):
        singleton_kernels = [kernel for kernel in kernel_list if len(kernel.active_dims) is 1]
        non_singleton_kernels = [kernel for kernel in kernel_list if len(kernel.active_dims) > 1]
        if len(non_singleton_kernels) == 0:
            return singleton_kernels
        rand_ind = self.RandomState.randint(len(non_singleton_kernels))
        kernel_to_split = non_singleton_kernels[rand_ind]
        old_active_dims = kernel_to_split.active_dims
        split_vec = self.RandomState.rand(len(old_active_dims)) < 0.5
        while (np.sum(split_vec) == 0 or np.sum(np.logical_not(split_vec)) == 0):
            split_vec = self.RandomState.rand(len(old_active_dims)) < 0.5 # make sure there is a split
        kernel1 = self.kernel(input_dim=np.sum(split_vec),
                               active_dims=old_active_dims[split_vec],
                               ARD=self.ARD, name=gen_name()) # default: var=1, lengthscales=1
        kernel2 = self.kernel(input_dim=np.sum(np.logical_not(split_vec)),
                               active_dims=old_active_dims[np.logical_not(split_vec)],
                               ARD=self.ARD, name=gen_name())
        if self.lengthscale is not None:
            kernel1.lengthscale.constrain_fixed(self.lengthscale)
            kernel2.lengthscale.constrain_fixed(self.lengthscale)
        # prob split: 1/|non_singleton_kernels| * 1/(2^(|dims_in_splitted_kernel|-1) - 1)
        proposal_logprob = -np.log(len(non_singleton_kernels)) -np.log(2**(len(old_active_dims)-1)-1)
        non_singleton_kernels.pop(rand_ind)
        non_singleton_kernels.append(kernel1)
        non_singleton_kernels.append(kernel2)
        new_kernel_list = non_singleton_kernels + singleton_kernels
        if self.variance is not None:
            for kernel in new_kernel_list:
                kernel.variance.constrain_fixed(self.variance)
        # prob merge: 1 / from len(new_kernel_list) choose 2
        kk = len(new_kernel_list)
        reverse_logprob = - np.log(kk) - np.log(kk-1) + np.log(2)
        return new_kernel_list, proposal_logprob, reverse_logprob

    def merge_kernel_random(self, kernel_list):
        if len(kernel_list) == 1:
            return kernel_list
        rand_ind1 = self.RandomState.randint(len(kernel_list))
        remaining = list(range(len(kernel_list)))
        remaining.pop(rand_ind1)
        rand_ind2 = self.RandomState.choice(remaining)
        kk = len(kernel_list)
        proposal_logprob = - np.log(kk) - np.log(kk-1) + np.log(2)
        old_kernel1 = kernel_list[rand_ind1]
        old_kernel2 = kernel_list[rand_ind2]
        new_active_dims = np.concatenate((old_kernel1.active_dims,old_kernel2.active_dims))
        new_kernel = self.kernel(input_dim=len(new_active_dims),
                                  active_dims=new_active_dims, ARD=self.ARD,
                                  name=gen_name())
        if self.lengthscale is not None:
            new_kernel.lengthscale.constrain_fixed(self.lengthscale)
        new_kernel_list = [v for i,v in enumerate(kernel_list) if i not in [rand_ind1,rand_ind2]]
        new_kernel_list.append(new_kernel)
        if self.variance is not None:
            for kernel in new_kernel_list:
                kernel.variance.constrain_fixed(self.variance)
        non_singleton_kernels = [kernel for kernel in new_kernel_list if len(kernel.active_dims) > 1]
        reverse_logprob = -np.log(len(non_singleton_kernels)) -np.log(2**(len(new_active_dims)-1)-1)
        return new_kernel_list, proposal_logprob, reverse_logprob

    def propose(self, kernel_list):
        non_singleton_kernels = [kernel for kernel in kernel_list if len(kernel.active_dims) > 1]
        if len(non_singleton_kernels) == 0:
            return self.merge_kernel_random(kernel_list)
        elif len(kernel_list) == 1:
            return self.split_kernel_random(kernel_list)
        elif self.RandomState.rand() < 0.5:
            return self.merge_kernel_random(kernel_list)
        else:
            return self.split_kernel_random(kernel_list)

class GPSampler():
    """
    Base sampler for additive kernel discovery in Gaussian Process models.
    """
    def __init__(self, X, y, model, lik_noise_var=1, fix_lik_noise_var=False,
                 lengthscale=None, variance=None,
                 optimizer=None, messages=False, max_iters=1000,
                 logging=False, seed=117,
                 kernel=GPy.kern.RBF, ARD=False, bounds=None):
        self.RandomState = np.random.RandomState(seed)
        self.set_XY(X, y)
        self.bounds = bounds
        self.lik_nv = lik_noise_var # internally defaults to 1 if None
        self.fix_lik_nv = fix_lik_noise_var
        self.optimizer = optimizer # internally defaults to scipy lbfgsb
        self.messages = messages
        self.iters = max_iters
        self.logging = logging
        self.kernel = kernel
        self.lengthscale = lengthscale
        self.variance = variance
        self.ARD = ARD if lengthscale is None else False
        if lengthscale is None:
            if ARD:
                self.lengthscale_value = []
                self.lprior = []
                for l, u in bounds:
                    self.lengthscale_value.append((u- l) / 3.)
                    self.lprior.append(InverseGamma(*invgamma_solver((u-l)/100., u-l)))
                self.lengthscale_value = np.array(self.lengthscale_value)
            else:
                l = np.min(bounds)
                u = np.max(bounds)
                self.lengthscale_value = (u - l) / 3.
                self.lprior = InverseGamma(*invgamma_solver((u-l)/100., u-l))
        if variance is None:
            var_value = 100  #(np.max(np.abs(self.y)) / 3.)**2
            self.vprior = GPy.priors.Gamma.from_EV(100, 100**2)#GPy.priors.Gamma.from_EV(10, 100) #InverseGamma(*invgamma_solver(1e-2, 1e1))#GPy.priors.Gamma.from_EV(var_value, var_value**2) # !!!!!!!!!!!!!!!
        if not fix_lik_noise_var:
            self.lnvprior = GPy.priors.LogGaussian(0, 1)
        self.optimize_params = True if self.lengthscale is None or self.variance is None else False
        self.curr_model = model # the model is assumed to have been optimized
        self.models = []
        if self.logging:
            self.models.append(self.curr_model.copy())

    def set_XY(self, X, y):
        self.X = X
        self.y = y

    def set_model(self, kernel_list):
        if self.optimize_params:
            if self.variance is None:
                var_value = 100 / float(len(kernel_list)) #(np.max(np.abs(self.y)) / 3.)**2 / float(len(kernel_list))
            for kk in kernel_list:   
                if self.lengthscale is None:
                    if self.ARD:
                        kk.lengthscale = self.lengthscale_value[kk.active_dims]
                        for i, a in enumerate(kk.active_dims):
                            kk.lengthscale[[i]].set_prior(self.lprior[a], warning=False)
                    else:
                        kk.lengthscale = self.lengthscale_value
                        kk.lengthscale.set_prior(self.lprior, warning=False)
                    #kk.lengthscale.constrain_bounded(1e-4, 1e3)
                if self.variance is None:
                    kk.variance = var_value
                    kk.variance.set_prior(self.vprior, warning=False)
                    #kk.variance.constrain_bounded(1e-4, 1e3)
        mean_function = self.curr_model.mean_function
        if mean_function is not None:
            mean_function = mean_function.copy()
        model = GPy.models.GPRegression(self.X, self.y,
                                        GPy.kern.Add(kernel_list),
                                        noise_var=self.lik_nv,
                                        mean_function=mean_function)
        model.update_model(False)
        if self.fix_lik_nv: # if lik_nv is known
            model['Gaussian_noise.variance'].constrain_fixed(self.lik_nv)
        else:
            model['Gaussian_noise.variance'].set_prior(self.lnvprior, warning=False)
        model.update_model(True)
        if self.optimize_params:
            model.optimize(optimizer=self.optimizer, messages=self.messages,
                           max_iters=self.iters)
        return model

    def run(self, X, y, iters=1):
        raise NotImplementedError()

class MHNO(GPSampler):
    """
    Metropolis Hastings sampler for non-overlapping decompositions.
    """
    def __init__(self, X, y, model, lik_noise_var=1, fix_lik_noise_var=False,
                 lengthscale=None, variance=None, optimizer=None, messages=False,
                 max_iters=1000, logging=False, seed=117,
                 kernel=GPy.kern.RBF, ARD=False, bounds=None,
                 proposal=None, force_max_lik=False):
        super(MHNO, self).__init__(X, y, model, lik_noise_var, fix_lik_noise_var,
                                   lengthscale, variance, optimizer, messages,
                                   max_iters, logging, seed,
                                   kernel, ARD, bounds)
        self.proposal = SplitMergeProposal(kernel, lengthscale, variance,
                                           ARD if lengthscale is None else False,
                                           self.RandomState) if not proposal else proposal
        self.alphas = []
        self.force_max_lik = force_max_lik

    def sample(self):
        return self.proposal.propose(self.curr_model.kern.copy().parts)

    def run(self, X, y, iters=1, verbose=False):
        self.set_XY(X, y)
        for i in range(iters):
            #print("In {} out of {}".format(i+1, iters))
            new_kernel_list, proposal_logprob, reverse_logprob = self.sample()
            new_model = self.set_model(new_kernel_list)
            new_loglik = new_model.log_likelihood()
            top = new_loglik + reverse_logprob
            bot = self.curr_model.log_likelihood() + proposal_logprob
            alpha = min(1, np.exp(top - bot))
            accept = self.RandomState.rand() <= alpha
            if verbose:
                print("{} {} {} {}".format(i, alpha, accept, [kk.active_dims for kk in new_model.kern.parts]))
            if accept and (not self.force_max_lik or new_loglik > self.curr_model.log_likelihood()):
                self.curr_model = new_model
        self.alphas.append(alpha)
        if self.logging:
            self.models.append(self.curr_model.copy())
        return self.curr_model

class GibbsNO(GPSampler):
    """
    Gibbs sampler for non-overlapping decompositions.
    """
    def __init__(self, X, y, model, lik_noise_var=1, fix_lik_noise_var=False,
                 lengthscale=None, variance=None,
                 optimizer=None, messages=False, max_iters=1000,
                 logging=False, seed=117,
                 kernel=GPy.kern.RBF, ARD=False, bounds=None,
                 dim_limit=None, force_max_lik=False):
        super(GibbsNO, self).__init__(X, y, model, lik_noise_var, fix_lik_noise_var,
                                     lengthscale, variance, optimizer, messages,
                                     max_iters, logging, seed,
                                     kernel, ARD, bounds)
        if dim_limit is not None:
            assert(dim_limit <= X.shape[1])
        self.dim_limit = dim_limit
        self.force_max_lik = force_max_lik

    def sample(self, iters):
        dim = self.X.shape[1]
        curr_model = self.curr_model
        decomp = [kernel.active_dims for kernel in curr_model.kern.parts]
        z = np.zeros(dim, dtype=int)
        for ii, dd in enumerate(decomp):
            z[np.array(dd)] = ii

        components = self.RandomState.choice(dim, dim, replace=False)
        counter = 0
        for d in components:
            len_p = np.sum(z[d] == z) 
            sep_model = curr_model if len_p == 1 else None
            log_prob = np.zeros(dim)
            llik = np.zeros(dim)
            new_models = []
            z0 = z[d]
            for m in range(dim):
                z[d] = m
                len_p = np.sum(z == m)
                # avoid unnecessary model optimizations if possible
                if z0 == m: 
                    new_model = curr_model
                elif len_p == 1 and sep_model is not None:
                    new_model = sep_model
                elif self.dim_limit is not None and len_p > self.dim_limit:
                    new_model = None
                    llik[m] = -np.inf
                else:
                    kernel_list = []
                    for v in np.unique(z):
                        active = np.where(z==v)[0]
                        kernel = self.kernel(input_dim=len(active),
                                           active_dims=active, ARD=self.ARD, 
                                           name=gen_name())
                        if self.lengthscale:
                            kernel.lengthscale.constrain_fixed(self.lengthscale)
                        if self.variance:
                            kernel.variance.constrain_fixed(self.variance)
                        kernel_list.append(kernel)
                    new_model = self.set_model(kernel_list)
                    counter += 1
                    if len_p == 1:
                        sep_model = new_model
                if self.dim_limit is None or len_p <= self.dim_limit:
                    llik[m] = new_model.log_likelihood()
                log_prob[m] = llik[m] + np.log(len_p -1 + 1)
                new_models.append(new_model)
            m = np.argmax(log_prob + self.RandomState.gumbel(size=dim))
            curr_model = new_models[m]
            if counter >= iters:
                break
        accept_llik = llik[m] > self.curr_model.log_likelihood()
        return curr_model, accept_llik, counter

    def run(self, X, y, iters=1, verbose=False):
        self.set_XY(X, y)
        i = 0
        dim = self.X.shape[1]
        while i < iters:
            gibbsiters = min(dim, iters-i)
            new_model, accept_llik, gibbsiters = self.sample(gibbsiters)
            if verbose:
                print("{} {}".format(i, [kk.active_dims for kk in new_model.kern.parts]))
            if not self.force_max_lik or accept_llik: # force_max_lik enabled: accept llik if increases
                self.curr_model = new_model
            i += gibbsiters
        if self.logging:
            self.models.append(self.curr_model.copy())
        return self.curr_model



    # def sample(self):
    #     dim = self.X.shape[1]
    #     curr_model = self.curr_model
    #     decomp = [kernel.active_dims for kernel in curr_model.kern.parts]
    #     z = np.zeros(dim, dtype=int)
    #     for ii, dd in enumerate(decomp):
    #         z[np.array(dd)] = ii
    #     for d in range(dim): # for each dimension
    #         len_p = np.sum(z[d] == z) 
    #         sep_model = curr_model if len_p == 1 else None
    #         log_prob = np.zeros(dim)
    #         llik = np.zeros(dim)
    #         new_models = []
    #         z0 = z[d]
    #         for m in range(dim):
    #             z[d] = m
    #             len_p = np.sum(z == m)
    #             # avoid unnecessary model optimizations if possible
    #             if z0 == m: 
    #                 new_model = curr_model
    #             elif len_p == 1 and sep_model is not None:
    #                 new_model = sep_model
    #             elif self.dim_limit is not None and len_p > self.dim_limit:
    #                 new_model = None
    #                 llik[m] = -np.inf
    #             else:
    #                 kernel_list = []
    #                 for v in np.unique(z):
    #                     active = np.where(z==v)[0]
    #                     kernel = self.kernel(input_dim=len(active),
    #                                        active_dims=active, ARD=self.ARD, 
    #                                        name=gen_name())
    #                     if self.lengthscale:
    #                         kernel.lengthscale.constrain_fixed(self.lengthscale)
    #                     if self.variance:
    #                         kernel.variance.constrain_fixed(self.variance)
    #                     kernel_list.append(kernel)
    #                 new_model = self.set_model(kernel_list)
    #                 if len_p == 1:
    #                     sep_model = new_model
    #             if self.dim_limit is None or len_p <= self.dim_limit:
    #                 llik[m] = new_model.log_likelihood()
    #             log_prob[m] = llik[m] + np.log(len_p -1 + 1)
    #             new_models.append(new_model)
    #         m = np.argmax(log_prob + self.RandomState.gumbel(size=dim))
    #         curr_model = new_models[m]
    #     accept_llik = llik[m] > self.curr_model.log_likelihood()
    #     return curr_model, accept_llik