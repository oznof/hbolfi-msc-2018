import numpy as np
import sobol_seq
import ghalton
import GPy
import scipy.stats as ss
import scipy.linalg as sl
from acquisition import LCBSC, LCBSCADD
from no_decomp import MHNO, GibbsNO
from utils import numgrad, mv_t, minimize, min_gp, invgamma_solver
from mcmc import metropolis, nuts, eff_sample_size, gelman_rubin
from priors import InverseGamma, Exponential, HalfT
import numdifftools as ndt
import time

class GPOpt:
    def __init__(self, f, kernel_list, bounds, acq_f=LCBSCADD, prior_f=None,
                 initial_n=10, mean_function=None,
                 lik_noise_var=1, fix_lik_noise_var=False,
                 lengthscale=None, variance=None,
                 optimizer=None, messages=False, max_iters=1000,
                 logging=False, seed=117, n_update=10,
                 ARD=False,
                 kernel_update=False, kernel_sampler=MHNO, 
                 kernel_sampler_kws={},
                 acq_f_kws={"noise_var":0.0}, 
                 xopt=None, f_true=None, additive=True, scaled=False):
        self.additive = additive
        self.normalize = False
        self.f = f
        self.bounds = bounds
        self.lb, self.ub = [np.array(b) for b in zip(*bounds)]
        self.dims = len(bounds)
        self.X = None
        self.y = None
        self.ARD = ARD
        self.prior_f = prior_f
        self.initial_n = initial_n
        self.update_initial_points()
        self.lik_nv = lik_noise_var # internally defaults to 1 if None
        self.fix_lik_nv = fix_lik_noise_var
        self.optimizer = optimizer # internally defaults to scipy lbfgsb
        self.messages = messages
        self.iters = max_iters
        self.logging = logging
        self.seed = seed
        self.n_update = n_update
        self.kernel_update = kernel_update
        self.optimize = True if lengthscale is None and variance is None else False
        self.lengthscale = lengthscale
        self.variance = variance 
        self.kernel_times = []
        self.acq_times = []
        self.scaled = scaled
        if lengthscale is None:
            if ARD:
                self.lengthscale_value = []
                self.lprior = []
                for l, u in bounds:
                    self.lengthscale_value.append((u- l) / 3.)
                    self.lprior.append(InverseGamma(*invgamma_solver((u-l)/100., u-l)))
                self.lengthscale_value = np.array(self.lengthscale_value)
                for kk in kernel_list:
                    kk.lengthscale = self.lengthscale_value[kk.active_dims]
                    for i, a in enumerate(kk.active_dims):
                        kk.lengthscale[[i]].set_prior(self.lprior[a])
            else:
                l = np.min(bounds)
                u = np.max(bounds)
                self.lengthscale_value = (u - l) / 3.
                self.lprior = InverseGamma(*invgamma_solver((u-l)/100., u-l))
                for kk in kernel_list:
                    kk.lengthscale = self.lengthscale_value
                    kk.lengthscale.set_prior(self.lprior)
        if variance is None:
            if not self.scaled:
                var_value =  100#(np.max(np.abs(self.y)) / 3.)**2
                self.vprior = GPy.priors.Gamma.from_EV(100, 100**2)#GPy.priors.Gamma.from_EV(10, 100) #InverseGamma(*invgamma_solver(1e-2, 1e1)) #GPy.priors.Gamma.from_EV(var_value, var_value) #!!!!!!!!!!!!!!!!!
            else:
                var_value = 0.1
                self.vprior = HalfT(1, 3)
            for kk in kernel_list:
                kk.variance = var_value
                kk.variance.set_prior(self.vprior, warning=False)
        if not fix_lik_noise_var:
            self.lnvprior = GPy.priors.LogGaussian(0, 1)
        self.curr_model = self.optimize_model(kernel_list, mean_function)
        if kernel_update:
            self.kernel_sampler = kernel_sampler(self.X, self.y, self.curr_model,
                                                 lik_noise_var, fix_lik_noise_var,
                                                 lengthscale, variance,
                                                 optimizer,messages, max_iters,
                                                 seed=seed, ARD=ARD, bounds=bounds,
                                                  **kernel_sampler_kws)
        if self.normalize:
            bounds = [_ for _ in zip(*(np.zeros(len(bounds)), np.ones(len(bounds))))]
        self.acq_f = acq_f(self.curr_model, bounds, max_opt_iters=max_iters,
                           seed=seed, **acq_f_kws)

        self.acq_ninit = acq_f_kws.pop("n_inits", max(10, int(len(bounds))))
        self.it = 0
        self.compute_regret = False if xopt is None or f_true is None else True
        if self.compute_regret:
            self.xopt = xopt
            self.f_true = f_true
            self.fopt = f_true(xopt)
            self.fbest = np.inf
            self.xbest = None
            self.xcurr = None
            self.expected_fcurr = None
            self.regret = None
            self.cumregret = 0
            self.update_regret() # find xcurr and xbest, but dont include it in cumregret
            self.cumregret = 0

    def update_regret(self):
        guess_points = np.vstack((self.xbest, self.xcurr)) if self.xcurr is not None else None
        x = min_gp(self.curr_model, additive=self.additive, bounds=self.bounds, 
                  seed=self.seed+self.it, guess_point=guess_points, 
                  minimizer_kws={"n_start_points": self.acq_ninit})
        self.expected_fcurr = self.curr_model.predict_noiseless(np.atleast_2d(x))[0].ravel()[0]
        self.xcurr = x
        fcurr = self.f_true(x)
        if fcurr < self.fbest:
            self.xbest = x
            self.fbest = fcurr
        self.regret = np.asscalar(fcurr - self.fopt)
        self.cumregret += self.regret

    def unscale(self, X):
        return (self.ub - self.lb)[None,:] * X + self.lb[None,:]

    def scale(self, X):
        return (X - self.lb[None,:]) / (self.ub - self.lb)[None,:]

    def update_initial_points(self):
        if self.prior_f is not None:
            self.X = np.vstack([self.prior_f.rvs() for _ in self.initial_n])
        else:
            if self.dims <= 40:
                self.X = sobol_seq.i4_sobol_generate(self.dims, self.initial_n)
            else:
                sequencer = ghalton.Halton(self.dims)
                self.X = np.array(sequencer.get(self.initial_n))
            self.X = self.unscale(self.X)
        self.y = np.hstack([self.f(x) for x in self.X])[:, None]
        if self.normalize:
            self.X = self.scale(self.X)

    def optimize_model(self, kernel_list, mean_function):
        #for kk in kernel_list:
        #    kk.lengthscale.constrain_bounded(1e-4, 1e4)
        #    kk.variance.constrain_bounded(1e-4, 1e4)
        model = GPy.models.GPRegression(self.X, self.y,
                                        GPy.kern.Add(kernel_list),
                                        noise_var=self.lik_nv,
                                        mean_function=mean_function)
        model.update_model(False)
        if self.fix_lik_nv: # if lik_nv is known
            model['Gaussian_noise.variance'].constrain_fixed(self.lik_nv)
        else:
            model['Gaussian_noise.variance'].set_prior(self.lnvprior)
        model.update_model(True)
        model.optimize(optimizer=self.optimizer, messages=self.messages,
                       max_iters=self.iters)
        return model

    def update_next_point(self, n=1, kernel_n_iters=10):
        self.it += 1
        if not (self.it % self.n_update): # optimize structure / hypers
            if self.optimize:
                self.curr_model.optimize(optimizer=self.optimizer,
                                         messages=self.messages,
                                         max_iters=self.iters)
            if self.kernel_update:
                st = time.perf_counter()
                self.curr_model = self.kernel_sampler.run(self.X, self.y, kernel_n_iters)
                self.kernel_times.append(time.perf_counter() - st)
        self.acq_f.set_model(self.curr_model)
        st = time.perf_counter()
        x = self.acq_f.acquire(n=n, t=self.it-1)
        self.acq_times.append(time.perf_counter() - st)
        if self.compute_regret and not (self.it % self.n_update):
            self.update_regret() # cumregret no longer makes sense if n_update > 1
                                 # doing this due to performance issues
        self.X = np.vstack((self.X, x))
        if self.normalize:
            x = self.unscale(x)
        y = [self.f(xx).ravel()[0] for xx in x]
        self.y = np.vstack((self.y, y))
        self.curr_model.set_XY(self.X, self.y)

class BOLFI(GPOpt):
    def unnormalized_loglikelihood_quantiles(self, threshold, x,
                                             quantiles=(0.025, 0.975),
                                             model=None, bounds=None,
                                             lik_noise_var=None):
        bounds = self.bounds if model is None else bounds
        model = self.curr_model if model is None else model
        lik_nv = self.lik_nv if lik_noise_var is None else lik_noise_var
        if bounds is None:
            raise ValueError("Must provide bounds.")
        x = np.asanyarray(x)
        x = x.reshape((-1, model.input_dim))
        logi = self._within_bounds(x, bounds=bounds)
        x = x[logi, :]
        if len(x) == 0:
            raise ValueError("All inputs outside bounds.")
        mean, var = model.predict_noiseless(x)
        sd = np.sqrt(var)
        sdn = np.sqrt(lik_nv)
        llogq = []
        idx = np.where(logi)[0]
        for alpha in quantiles:
            logqi = -np.ones(len(x)) * np.inf
            logqi[idx] = ss.norm.logcdf((sd * ss.norm.ppf(alpha) - mean +
                                         threshold) / sdn).ravel()
            llogq.append(logqi)
        return llogq


    def unnormalized_loglikelihood(self, threshold, x, model=None, bounds=None):
        bounds = self.bounds if model is None else bounds
        model = self.curr_model if model is None else model
        if bounds is None:
            raise ValueError("Must provide bounds.")
        x = np.asanyarray(x)
        ndim = x.ndim
        x = x.reshape((-1, model.input_dim))
        logpdf = -np.ones(len(x)) * np.inf
        logi = self._within_bounds(x, bounds=bounds)
        x = x[logi, :]
        if len(x) == 0:
            if ndim == 0 or (ndim == 1 and model.input_dim > 1):
                logpdf = logpdf[0]
            return logpdf
        mean, var = model.predict(x)
        logpdf[np.where(logi)[0]] = ss.norm.logcdf(threshold, mean, np.sqrt(var)).ravel()
        if ndim == 0 or (ndim == 1 and model.input_dim > 1):
            logpdf = logpdf[0]
        return logpdf

    def find_map(self, threshold, model=None, bounds=None, seed=None, guess_point=None,
                 eps=1e-20):
        random_state = np.random.RandomState(seed) if seed else np.random
        bounds = self.bounds if model is None else bounds
        model = self.curr_model if model is None else model
        print(threshold)
        f = lambda x: -self.logpdf(threshold, x, model, bounds) + eps
        grad_f = lambda x: -self.gradient_logpdf(threshold, x, model, bounds)
        xmap  = minimize(f, bounds, grad=grad_f, random_state=random_state,
                         guess_point=guess_point)[0]
        return xmap


    def gradient_unnormalized_loglikelihood(self, threshold, x, model=None,
                                            bounds=None, eps=1e-20):
        bounds = self.bounds if model is None else bounds
        model = self.curr_model if model is None else model
        if bounds is None:
            raise ValueError("Must provide bounds.")
        x = np.asanyarray(x)
        ndim = x.ndim
        x = x.reshape((-1, model.input_dim))
        grad = np.zeros_like(x)
        logi = self._within_bounds(x, bounds=bounds)
        x = x[logi, :]
        if len(x) == 0:
            if ndim == 0 or (ndim == 1 and model.input_dim > 1):
                grad = grad[0]
            return grad
        mean, var = model.predict(x)
        var += eps
        std = np.sqrt(var)
        grad_mean, grad_var = model.predictive_gradients(x)
        if grad_mean.ndim == 3:
            grad_mean = grad_mean[:, :, 0] # Assume 1D output (distance in ABC)
        factor = -grad_mean * std - (threshold - mean) * 0.5 * grad_var / std
        factor = factor / var
        term = (threshold - mean) / std
        pdf = ss.norm.pdf(term)
        cdf = ss.norm.cdf(term)
        grad[logi, :] = factor * np.round(pdf / (cdf + eps), 20)
        if ndim == 0 or (ndim == 1 and model.input_dim > 1):
            grad = grad[0]
        return grad

    def hessian_nloglikelihood2(self, threshold, x, model=None,
                                           bounds=None, eps=1e-8):
        x = np.atleast_2d(x)
        #f_grad = lambda x: -self.gradient_unnormalized_loglikelihood(threshold,
        #                                        x, model=model, bounds=bounds)
        #hessians = [numgrad(f_grad, xx, hessian=True) for xx in x]
        hessians = [ndt.Hessian(lambda x: -self.unnormalized_loglikelihood(threshold,
                                x, model=model, bounds=bounds))(xx) for xx in x]
        if x.shape[0] == 1:
            hessians = hessians[0]
        return hessians

    def hessian_nloglikelihood(self, threshold, x, model=None, bounds=None, eps=1e-20):
        bounds = self.bounds if model is None else bounds
        model = self.curr_model if model is None else model
        if bounds is None:
            raise ValueError("Must provide bounds.")
        x = np.asanyarray(x)
        x = x.reshape((-1, model.input_dim))
        assert(x.shape[0] == 1) # only accept one point for now
        assert(self._within_bounds(x, bounds=bounds))
        mean, var = [_.ravel()[0] for _ in model.predict(x)]
        var += eps
        std = np.sqrt(var)
        grad_mean, grad_var = model.predictive_gradients(x)
        if grad_mean.ndim == 3:
            grad_mean = grad_mean[:, :, 0]
        z = (threshold - mean) / std
        cdf = ss.norm.cdf(z)
        pdf = ss.norm.pdf(z)
        dz_dx = -grad_mean / std - .5 * z * grad_var / var
        r = pdf / (cdf + eps)
        #H_mean, H_var = \
        #[numgrad(lambda x: model.predictive_gradients(np.atleast_2d(x))[i].ravel(),
        #         x[0], hessian=True) for i in range(2)]  # has numerical problems
        H_mean = ndt.Hessian(lambda x: np.asscalar(model.predict_noiseless(np.atleast_2d(x))[0]))(x[0])
        H_var = ndt.Hessian(lambda x: np.asscalar(model.predict_noiseless(np.atleast_2d(x))[1]))(x[0])
        t1 = (z + r) * dz_dx.T * dz_dx
        H_z_t1 = - H_mean / std + .5 / (std * var) * grad_var.T * grad_mean
        H_z_t2 = - 0.5 * (-grad_mean.T * grad_var / (std * var) \
                          -1.5 * z / var**2 * grad_var.T * grad_var \
                          + z / var * H_var)
        H = r * (t1 - H_z_t1 - H_z_t2)
        return H

    def laplace_aprx(self, threshold, model=None, bounds=None, seed=None,
                     guess_point=None, df=None):
        bounds = self.bounds if model is None else bounds
        model = self.curr_model if model is None else model
        if bounds is None:
            raise ValueError("Must provide bounds.")
        xstar = self.find_map(threshold, model=model, bounds=bounds, seed=seed,
                              guess_point=guess_point)
        H = self.hessian_nloglikelihood(threshold, xstar, model, bounds)
        H[np.diag_indices(len(H))] += 1e-8
        try:
            C = sl.cho_solve(sl.cho_factor(H), np.eye(xstar.size))
        except np.linalg.LinAlgError:
            print("ERROR (possibly in the boundary): ")
            print("xstar: ", xstar)
            print("=================")
            print(H)
            print(self.hessian_nloglikelihood2(threshold, xstar, model, bounds))
            try:
                H = self.hessian_nloglikelihood2(threshold, xstar, model, bounds)
                H[np.diag_indices(len(H))] += 1e-8
                C = sl.cho_solve(sl.cho_factor(H), np.eye(xstar.size))
            except:
                return None
        if df is None:
            sampler = ss.multivariate_normal(mean=xstar, cov=C)
            sampler.random_state = np.random if seed is None else np.random.RandomState(seed)
        else:
            sampler = mv_t(mean=xstar, cov=C, df=df, seed=seed) # multivariate t
        return sampler

    def logpdf(self, threshold, x, model=None, bounds=None):
        """
        Unnormalized log-posterior pdf at x.
        """
        bounds = self.bounds if model is None else bounds
        model = self.curr_model if model is None else model
        if bounds is None:
            raise ValueError("Must provide bounds.")
        prior_logpdf = -np.sum(np.log(self.ub - self.lb)) if self.prior_f is \
                        None else self.prior_f.logpdf(x)
        logpdf = self.unnormalized_loglikelihood(threshold, x, model, bounds) +\
                 prior_logpdf
        return logpdf

    def gradient_logpdf(self, threshold, x, model=None, bounds=None):
        """
        Gradient of the unnormalized log-posterior pdf at x.
        """
        bounds = self.bounds if model is None else bounds
        model = self.curr_model if model is None else model
        prior_gradient_logpdf = 0 if self.prior_f is None else\
                                    self.prior_f.gradient_logpdf(x)
        grads = self.gradient_unnormalized_loglikelihood(threshold, x,
                                                         model, bounds) + \
                prior_gradient_logpdf
        # nan grads are result from -inf logpdf
        # return np.where(np.isnan(grads), 0, grads)
        return grads

    def _within_bounds(self, x, model=None, bounds=None):
        model = self.curr_model if model is None else model
        bounds = self.bounds if bounds is None else bounds
        x = x.reshape((-1, model.input_dim))
        logical = np.ones(len(x), dtype=bool)
        for i in range(self.dims):
            logical *= (x[:, i] >= bounds[i][0])
            logical *= (x[:, i] <= bounds[i][1])
        return logical

    def mcmc(self, threshold, n_iter=10000, model=None, bounds=None,
             params0=None, seed=None, get_stats=True, n_chains=1, 
             sampler="nuts", thinning=1, sigma_proposals=None, **kws):
        bounds = self.bounds if model is None else bounds
        model = self.curr_model if model is None else model
        if params0 is None:
            params0 = self.find_map(threshold, model=model,
                                    bounds=bounds, seed=seed)
        f = lambda x: self.logpdf(threshold, x, model, bounds)
        f_grad = lambda x: self.gradient_logpdf(threshold, x, model, bounds)
        if sampler == "nuts":
            if n_chains > 1:
                from joblib import Parallel, delayed
                sampler = lambda i: nuts(n_iter, params0, f, f_grad, seed=seed+i, **kws)
                samples = np.stack(Parallel(n_jobs=n_chains)(delayed(sampler)(i) for i in range(n_chains)))
            else:
                samples = nuts(n_iter, params0, f, f_grad, seed=seed, **kws)[None, :, :]
        elif sampler == "mh": # only support 1 chain for now (joblib should be replaced with something else)
            assert(n_chains == 1)
            if sigma_proposals is None:
                sigma_proposals = 1e-2
            samples = metropolis(n_iter, params0, f, sigma_proposals=sigma_proposals, seed=seed)[None, :, :]
        samples = samples[:, ::thinning, :]
        if get_stats:
            eff_sizes = np.array([eff_sample_size(samples[:, :, i]) for i in range(samples.shape[-1])])
            psrf = np.array([gelman_rubin(samples[:, :, i]) for i in range(samples.shape[-1])])
        samples = samples.squeeze()
        return (samples, eff_sizes, psrf) if get_stats else samples

