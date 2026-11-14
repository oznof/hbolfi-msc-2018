import numpy as np
import math
import scipy
import scipy.stats as ss
import scipy.linalg as sl
from scipy.special import gammaln
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import distance_matrix
import scipy.integrate as integrate
from qpoly import QPoly, LogQPoly, QuarticPoly
import ghalton
import sobol_seq
import scipydirect
from scipy.stats._multivariate import _PSD
import time
import datetime
import itertools
import string
import random as rnd
from sympy.solvers import solve
from sympy import Symbol

def gen_name():
    return rnd.choice(string.ascii_letters) + rnd.choice(string.ascii_letters)

def reset_kernel_hypers(kernels):
    for kernel in kernels:
        kernel.lengthscale = 1
        kernel.variance = 1

class timer():
    def __init__(self, n_iters):
        self.curr_time = time.time()
        self.time_total = 0
        self.curr_iter = 0
        self.times = []
        self.n_iters = n_iters
        self.time_mean = np.nan
        self.eta = np.nan

    def update(self):
        elapsed = time.time() - self.curr_time
        self.times.append(elapsed)
        self.time_total += elapsed
        self.time_mean = np.mean(self.times)
        self.curr_iter +=1
        self.eta = self.time_mean * (self.n_iters - self.curr_iter)
        self.curr_time = time.time()
        return self.time2str(self.time_total), self.time2str(self.eta)

    def time2str(self, t):
        return str(datetime.timedelta(seconds=t))


def minimize(fun, bounds, grad=None, prior=None, n_start_points=None,
             maxiter=1000, random_state=None, options=None, guess_point=None):
    """Find the minimum of function 'fun'.
    Parameters
    ----------
    fun : callable
        Function to minimize.
    bounds : list of tuples
        Bounds for each parameter.
    grad : callable
        Gradient of fun or None.
    prior : scipy-like distribution object
        Used for sampling initialization points. If None, samples uniformly.
    n_start_points : int, optional
        Number of initialization points.
    maxiter : int, optional
        Maximum number of iterations.
    random_state : np.random.RandomState, optional
        Used only if no elfi.Priors given.
    Returns
    -------
    tuple of the found coordinates of minimum and the corresponding value.
    """
    if n_start_points is None:
        n_start_points = max(10, len(bounds))
    if options is None:
        options = {}
    options["maxiter"] = maxiter
    ndim = len(bounds)
    start_points = np.empty((n_start_points, ndim))
    if prior is None:
        random_state = random_state or np.random
        #for i in range(ndim):
        #    start_points[:, i] = random_state.uniform(*bounds[i], n_start_points)
        start_points = sobol_seq.i4_sobol_generate(ndim, n_start_points)
        start_points = (bounds[:, 1] - bounds[:, 0])[None,:] * start_points + bounds[:, 0][None,:]
    else:
        start_points = prior.rvs(n_start_points, random_state=random_state)
        if len(start_points.shape) == 1:
            # Add possibly missing dimension when ndim=1
            start_points = start_points[:, None]
        for i in range(ndim):
            start_points[:, i] = np.clip(start_points[:, i], *bounds[i])
    if guess_point is not None:
        start_points = np.vstack((np.atleast_2d(guess_point), start_points))
    # Run the optimisation from each initialization point.
    n_start_points = start_points.shape[0]
    locs = []
    vals = np.empty(n_start_points)
    for i in range(n_start_points):
        result = scipy.optimize.minimize(fun, start_points[i, :],
                                         method='L-BFGS-B', jac=grad, bounds=bounds,
                                         options=options)
        locs.append(result['x'])
        vals[i] = result['fun']
    # Return the optimal case.
    ind_min = np.argmin(vals)
    locs_out = locs[ind_min]
    for i in range(ndim):
        locs_out[i] = np.clip(locs_out[i], *bounds[i])
    return locs[ind_min], vals[ind_min]

def pred_noiseless(X, model, kernel=None):
    X = np.atleast_2d(X)
    if kernel is not None and X.shape[1] < model.input_dim:
        Z = np.zeros((X.shape[0], model.input_dim))
        Z[:, kernel.active_dims] = X
        X = Z
    mean, var = model.predict_noiseless(X, kern=kernel)
    return mean.squeeze(), var.squeeze()

def pred_grad(X, model, kernel=None):
    X = np.atleast_2d(X)
    if kernel is not None and X.shape[1] < model.input_dim:
        Z = np.zeros((X.shape[0], model.input_dim))
        Z[:, kernel.active_dims] = X
        X = Z
    grad_mean, grad_var = model.predictive_gradients(X, kern=kernel)
    grad_mean = grad_mean[:,:,0] if kernel is None else grad_mean[:, kernel.active_dims, 0]
    grad_var = grad_var if kernel is None else grad_var[:, kernel.active_dims]
    return grad_mean.squeeze(), grad_var.squeeze()

def min_gp(model, bounds, additive=True, X=None, minimizer="lbfgs", seed=None, 
    minimizer_kws={}, guess_point=None):
    minimizer_kws = minimizer_kws.copy()
    guess_point = np.atleast_2d(guess_point) if guess_point is not None else None
    if X is None:
        xmin = np.zeros(model.input_dim)
        if minimizer == "lbfgs":
            grad = minimizer_kws.pop("grad", True)
            minimizer_kws.setdefault("n_start_points", max(10, len(bounds)))
            minimizer_kws.setdefault("maxiter", 1000)
            minimizer_kws.setdefault("random_state", np.random.RandomState(seed) if seed else np.random)
        elif minimizer == "direct":
            minimizer_kws.setdefault("eps", 1e-4)
            minimizer_kws.setdefault("maxf", 20000)
            maxT = minimizer_kws.pop("maxT", None)
            minimizer_kws.setdefault("algmethod", 0)
    else:
        res = {}
    kernels = model.kern.parts if additive else [model.kern]
    if additive and isinstance(model.mean_function, (QPoly, LogQPoly, QuarticPoly)): # save state of mean_function
        orig = model.mean_function
        nparts = float(len(kernels))
    for kernel in kernels:
        active_dims = kernel.active_dims
        if additive and isinstance(model.mean_function, (QPoly, LogQPoly, QuarticPoly)):
            mask = np.zeros(model.input_dim, dtype=bool)
            mask[active_dims] = True
            model.mean_function = orig.copy()
            model.mean_function.A = model.mean_function.A.values * mask[:,None]
            model.mean_function.B = model.mean_function.B.values * mask[:,None]
            if isinstance(model.mean_function, QuarticPoly):
                model.mean_function.C = model.mean_function.C.values * mask[:,None]
                model.mean_function.D = model.mean_function.D.values / nparts
            else:
                model.mean_function.C = model.mean_function.C.values / nparts
        f_d = lambda X: pred_noiseless(X, model, kernel)[0]
        grad_d = lambda X: pred_grad(X, model, kernel)[0]
        if X is not None:
            res[str(active_dims.tolist())] = (f_d(X), grad_d(X))
        else:
            if minimizer == "lbfgs":
                grad_d1 = grad_d if grad else None
                x0 = guess_point[:, active_dims] if guess_point is not None else None
                xmin[active_dims] = minimize(f_d, bounds[active_dims], grad=grad_d1,
                                             guess_point=x0, **minimizer_kws)[0]
            elif minimizer == "direct":
                maxT1 = min(5000, 100*len(active_dims)) if maxT is None else maxT
                f_d1 = lambda x, user_data: f_d(x)
                xmin[active_dims] = scipydirect.minimize(f_d1, bounds=bounds[active_dims], maxT=maxT1,
                                                 **minimizer_kws)[0]
            else:
                raise ValueError("Minimizer not supported.")
    if additive and isinstance(model.mean_function, (QPoly, LogQPoly, QuarticPoly)): # recover state
        model.mean_function = orig
    return xmin if X is None else res


def integrate_f(f, bounds):
    bounds = np.atleast_2d(bounds)
    ndim = bounds.shape[0]
    if ndim == 1:
        v = integrate.quad(f, bounds[0][0], bounds[0][1])[0]
    elif ndim == 2:
        f1 = lambda y, x: f(np.array([x, y]))
        v = integrate.dblquad(f1, bounds[0][0], bounds[0][1],
                              lambda x: bounds[1][0], lambda x: bounds[1][1])[0]
    elif ndim == 3:
        f1 = lambda z, y, x: f(np.array([x, y, z]))
        v = integrate.tplquad(f1, bounds[0][0], bounds[0][1],
                              lambda x: bounds[1][0], lambda x: bounds[1][1],
                              lambda x, y: bounds[2][0], lambda x, y: bounds[2][1])[0]
    else:
        raise ValueError("Number of dimensions not supported.")
    return v

def normalize(f, bounds, return_Z=False):
        Z = integrate_f(f, bounds)
        g = lambda x: f(x) / Z
        return (g, Z) if return_Z else f

def dist(p, q, bounds, measure="kl", eps=1e-20):
    if measure == "kl":
        f = lambda x: p(x) * (np.log(p(x) + eps) - np.log(q(x) + eps))
    elif measure == "tv":
        f = lambda x: 0.5 * np.abs(p(x) - q(x))
    else:
        raise ValueError("Measure not supported.")
    return integrate_f(f, bounds)

def kl_normal(mu0, cov0, mu1, cov1):
    L0 = sl.cholesky(cov0)
    L1 = sl.cholesky(cov1)
    mus = mu1.squeeze() - mu0.squeeze()
    kl01 = .5 * (np.sum(np.diag(sl.cho_solve((L1,False), cov0))) +  mus.T.dot(sl.cho_solve((L1,False), mus)) - mu0.size
                 + 2 * np.sum(np.log(np.diag(L1))) - 2 * np.sum(np.log(np.diag(L0))))
    return kl01

def kl_normal_sym(mu0, cov0, mu1, cov1):
    return 0.5 * (kl_normal(mu0, cov0, mu1, cov1) + kl_normal(mu1, cov1, mu0, cov0)) 

def partition_metrics(A, B, eps=1e-20):
    A = np.asanyarray(A)
    B = np.asanyarray(B)
    n = len(A)
    assert(len(B) == n)
    tp = eps
    tn = eps
    pred_p = eps
    for i in range(n-1):
        pred_p += np.sum(B[i] == B[i+1:])
        tp += np.sum(np.logical_and(A[i] == A[i+1:], B[i] == B[i+1:]))
        tn += np.sum(np.logical_and(A[i] != A[i+1:], B[i] != B[i+1:]))
    total_pairs = n * (n-1) / 2
    rand_idx = (tp + tn) / total_pairs
    truth_p = np.sum([(A[i] == A[i+1:]).sum() for i in range(n-1)]) + eps
    truth_n = np.sum([(A[i] != A[i+1:]).sum() for i in range(n-1)]) + eps
    tpr = tp / truth_p
    tnr = tn / truth_n
    prec = tp / pred_p
    f1_score = 2 / (1/tpr + 1/prec)
    return rand_idx, tpr, tnr, prec, f1_score

def numgrad(f, x, hessian=False, eps=1e-8):
    assert(x.ndim == 1)
    shape = (x.size, x.size) if hessian else x.size
    g = np.zeros(shape)
    for i in range(x.size):
        x[i] += eps/2
        f1 = f(x)
        #print(f1)
        x[i] -= eps
        #print((f1 - f(x)) / eps)
        g[i] = (f1 - f(x)) / eps
        #print(g)
        #asdasdasd
        x[i] += eps/2
    return g

def unscale(X, bounds):
    X = np.atleast_2d(X)
    return (bounds[:, 1] - bounds[:, 0])[None,:] * X + bounds[:, 0][None,:]

def mcint(f, bounds, n=100000, quasi=False, seed=None, nbatch=100000):
    assert(n % nbatch == 0)
    bounds = np.atleast_2d(bounds)
    ndim = bounds.shape[0]
    if quasi:
        halton = ghalton.Halton(ndim)
        sampler = lambda n: np.vstack(halton.get(n))
    else:
        RandomState = np.random.RandomState(seed) if seed else np.random
        sampler = lambda n: RandomState.uniform(size=(n, ndim))
    total = 0
    total_sq = 0
    for i in range(n // nbatch):
        fi = f(unscale(sampler(nbatch), bounds))
        total += np.sum(fi)
        total_sq += np.sum(fi**2)
    mean = total / n
    var = (total_sq - n * mean**2) / (n - 1)
    vol = np.prod(bounds[:, 1] - bounds[:, 0])
    I = vol * mean
    I_std = vol * np.sqrt(var / n)
    return I if quasi else (I, I_std)

def imp_samp(f, sampler, n=100000, nbatch=100000): # sampler must have methods rvs and pdf
    assert(n % nbatch == 0)
    total = 0
    total_sq = 0
    for i in range(n // nbatch):
        si = sampler.rvs(nbatch)
        fi = f(si) / sampler.pdf(si)
        total += np.sum(fi)
        total_sq += np.sum(fi**2)
    I = total / n
    I_std = np.sqrt(1/n * (total_sq / n - I**2))
    return I, I_std

def logsumexp(x):
    imax = np.argmax(x)
    return np.log(np.sum(np.exp(x - x[imax]))) + x[imax]

def sir(f, sampler, n=100000, steps=1, seed=None, rfactor=1,
        return_samples=False, eps=1e-20):
    assert(0 < rfactor <= 1)
    si = sampler.rvs(n)
    logpdf = sampler.logpdf(si)
    logfi = np.log(f(si) + eps)
    logwi = logfi - logpdf
    wi = np.exp(logwi - logsumexp(logwi))
    RandomState = np.random.RandomState(seed) if seed else np.random
    for i in range(steps):
        idx = RandomState.choice(len(si), size=len(si)*1,  replace=True, p=wi)
        si = si[idx]
        wi = np.ones(len(si)) / float(len(si))
    samples = (si, wi)
    f_eval = np.sum(f(si) * wi)
    return (f_eval, samples) if return_samples else f_eval

class mv_t:
    """
    Multivariate t-distribution.
    """
    def __init__(self, mean, cov, df, seed=None):
        self.mean = np.atleast_2d(mean)
        self.cov = cov
        psd = _PSD(cov, allow_singular=False)
        self.prec_U = psd.U
        self.log_det_cov = psd.log_pdet
        self.df = df
        self.p = cov.shape[0]
        self.RandomState = np.random.RandomState(seed) if seed else np.random

    def logpdf(self, x):
        """
        x : n x p
        """
        x = np.atleast_2d(x).reshape(-1, self.p)
        p = self.p
        df = self.df
        dev = x - self.mean
        maha = np.sum(np.square(np.dot(dev, self.prec_U)), axis=-1)
        logdist = -(df+p)/2. * np.log1p(maha / df)
        logZ =  gammaln((df+p)/2.) - gammaln(df/2.) - p/2. * (np.log(df) + np.log(np.pi)) \
                - 0.5 * self.log_det_cov
        return logZ + logdist

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def rvs(self, size):
        y = self.RandomState.multivariate_normal(mean=np.zeros(self.p),
                                                 cov=self.cov, size=size)
        u = self.RandomState.chisquare(self.df, size) / self.df
        return  self.mean + y / np.sqrt(u)[:,None]

def sq_mmd(xs, ys, wxs=None, wys=None, scale=None, return_scale=False,
           only_scale=False):
    """
    Square maximum mean discrepancy with gaussian kernel.
    """
    n_x = xs.shape[0]
    n_y = ys.shape[0]
    xx = pdist(xs, metric="sqeuclidean")
    if scale is None:
        scale = np.median(np.sqrt(xx)) if scale is None else scale
        if only_scale:
            return scale
    yy = pdist(ys, metric="sqeuclidean")
    xy = cdist(xs, ys, metric="sqeuclidean")

    c = -0.5 / (scale ** 2)
    if wxs is None:
        xx = np.sum(np.exp(c * xx)) / (n_x * (n_x - 1))
    else:
        wxs /= np.sum(wxs)
        wxx = np.hstack([wx * wxs[i+1:] for i, wx in enumerate(wxs)])
        xx = np.sum(wxx * np.exp(c * xx)) / (1.0 - np.sum(wxs ** 2))
    if wys is None:
        yy = np.sum(np.exp(c * yy)) / (n_y * (n_y - 1))
    else:
        wys /= np.sum(wys)
        wyy = np.hstack([wy * wys[i+1:] for i, wy in enumerate(wys)])
        yy = np.sum(wyy * np.exp(c * yy)) / (1.0 - np.sum(wys ** 2))
    if wxs is None and wys is None:
        xy = np.sum(np.exp(c * xy)) / (n_x * n_y)
    else:
        if wxs is None:
            wxs = np.full(n_x, 1.0 / n_x)
        if wys is None:
            wys = np.full(n_y, 1.0 / n_y)
        wxy = np.dot(wxs[:, None], wys[None, :])
        xy = np.sum(wxy * np.exp(c * xy)) / (n_x * n_y)
    mmd2 = 2 * ( xx + yy - xy) # xx, yy need to be accounted twice due to pdist
    return (mmd2, scale) if return_scale else mmd2

def bootstrap_test(xs, ys, wxs=None, wys=None,
                   measure="sqmmd", seed=None, n_bs=10, n_jobs=4):
    """
    Bootstrap.
    """
    n_x = xs.shape[0]
    n_y = ys.shape[0]
    if measure == "sqmmd":
        scale = sq_mmd(xs, ys, wxs, wys, only_scale=True)
        args = (scale,)
        def bs(xs, ys, wxs, wys, n_x, n_y, scale, seed=None):
            RandomState = np.random.RandomState(seed) if seed else np.random
            ix = RandomState.randint(0, n_x, n_x)
            iy = RandomState.randint(0, n_y, n_y)
            wxs = wxs[ix] if wxs is not None else wxs
            wys = wys[ix] if wys is not None else wys
            return sq_mmd(xs[ix], ys[iy], wxs, wys, scale=scale)
    elif measure == "qkl":
        args = ()
        def bs(xs, ys, wxs, wys, n_x, n_y, seed=None):
            RandomState = np.random.RandomState(seed) if seed else np.random
            ix = RandomState.randint(0, n_x, n_x)
            iy = RandomState.randint(0, n_y, n_y)
            wxs = wxs[ix] if wxs is not None  else wxs
            wys = wys[ix] if wys is not None  else wys
            return quasi_kl(xs[ix], ys[iy], wxs, wys)

    res = np.hstack(Parallel(n_jobs=n_jobs)(delayed(bs)(xs, ys, wxs, wys, n_x,
                                 n_y, *args, seed=seed+i if seed else seed) for i in range(n_bs)))
    mean_bs = np.mean(res)
    std_bs = np.std(res, ddof=1)
    return mean_bs, std_bs


def quasi_kl(xs, ys, wxs=None, wys=None, eps=1e-20):
    # THIS WONT WORK WELL
    xs = np.atleast_2d(xs)
    ys = np.atleast_2d(ys)
    n_x, dim = xs.shape
    n_y = ys.shape[0]
    xx = cdist(xs, xs)
    xx[np.diag_indices(n_x)] = np.inf
    xx = xx.min(1)
    xy = cdist(xs, ys).min(1)
    if wxs is None:
        wxs = 1. / n_x
    return dim * np.sum(wxs * (np.log(xy+eps) - np.log(xx+eps))) + np.log(n_y/(n_x-1))


def pilotABC(simulator, quantile=0.9, sampler=None, n_sample=5000,
             bounds=None, seed=None):
    if sampler is None: # uniform given bounds
        rngABC = np.random.RandomState(seed) if seed else np.random
        sampler = lambda : rngABC.uniform(low=bounds[:,0], high=bounds[:,1])
    dists = np.zeros(n_sample)
    for i in range(n_sample):
        x = sampler()
        dists[i] = simulator(x)
    return np.sort(dists)[int(quantile*n_sample)]

def rejectionABC(simulator, threshold=None, sampler=None, n_sample=100000,
                 bounds=None, seed=None, verbose=False):
    if sampler is None: # uniform given bounds
        rngABC = np.random.RandomState(seed) if seed else np.random
        sampler = lambda : rngABC.uniform(low=bounds[:,0], high=bounds[:,1])
    # simulator with discrepancy
    n_simulation = 0
    n_accepted = 0
    samples = []
    fvalues = []

    while n_accepted < n_sample:
        n_simulation += 1
        x = sampler()
        fx = simulator(x)
        if threshold is None or fx  <= threshold:
            samples.append(x)
            fvalues.append(fx)
            n_accepted += 1

        if verbose and n_simulation % 1000 == 0:
            print("rABC: Accepted {} out of {}.".format(n_accepted, n_simulation))
    samples = np.vstack(samples)
    fvalues = np.hstack(fvalues)
    return   samples, fvalues, n_accepted

def isampler(sampler, loglik, prior=None, n_sample=10000,
             seed=None, random_state=None, resample=False, 
             return_ess=False, ess_target=0.1, max_iters=1000, 
             factor=1):
    x = None#np.array([[]])
    w = None#np.array([])
    ess = 0
    i = 0
    while(ess < ess_target * n_sample or i < max_iters): #for i in range(100):
        xi = sampler.rvs(size=n_sample*factor, random_state=random_state)
        wi = loglik(xi) - sampler.logpdf(xi)
        if prior: # otherwise assume uniform
            wi += prior.logpdf(xi)
        if x is None:
            x = xi
            w = wi
        else:
            x = np.vstack((x, xi))
            w = np.hstack((w, wi))
            idx = np.argsort(-w)[:n_sample]
            x = x[idx]
            w = w[idx]
        ess = np.exp(2*logsumexp(w) -logsumexp(2 * w))
        print("i: {}   ess: {}".format(i, ess))
        i += 1
    w = np.exp(w - logsumexp(w))  
    if resample:
        if random_state is None:
            random_state = np.random if seed is None else np.random.RandomState(seed)
        out = (x[random_state.choice(n_sample, size=n_sample, p=w)],)
    else:
        out = (x, w)
    if return_ess:
        out +=  (ess,)
    return out


def SMCABC(simulator, threshold_init, threshold_last, threshold_decay=None,
           prior=None, n_sample=10000, bounds=None, seed=None,
           ess_min=0.5, verbose=False):
    rng = np.random.RandomState(seed) if seed else np.random
    if prior is None: # uniform given bounds
        sampler = lambda : rng.uniform(low=bounds[:,0], high=bounds[:,1])
    else:
        sampler = lambda : prior.rvs()
    prior_logpdf = (lambda x: -np.sum(np.log(bounds[:,1] - bounds[:,0]))) if prior is None \
                    else (lambda x: prior.logpdf(x))
    if threshold_decay is None:
        threshold_decay = (threshold_last - threshold_init) / 10
    log_n_sample = np.log(n_sample)
    log_ess_min = np.log(ess_min)
    log_ws = np.full(n_sample, -log_n_sample)
    iter = 0
    n_sim = 0
    threshold = threshold_init
    if verbose: print("threshold init:  ", threshold, "threshold last:  ",
                      threshold_last, "threshold_decay:  ", threshold_decay)
    samples, n_simulation = rejectionABC(simulator, threshold, sampler,
                                         n_sample, bounds=bounds,
                                         verbose=verbose)
    n_sim += n_simulation
    n_dim = samples.shape[1]
    while threshold > threshold_last:
        if verbose: print("iter: ", iter, "  simulations:   ", n_sim)
        iter += 1
        threshold += threshold_decay
        ws = np.exp(log_ws)
        # compute covariance
        cov = 2 * np.cov(samples, rowvar=False, ddof=0)
        std = np.linalg.cholesky(cov)

        new_samples = np.empty((n_sample, n_dim))
        new_log_ws = np.empty(n_sample)

        for i in range(n_sample):
            dist = np.inf
            while dist > threshold:
                idx = rng.choice(n_sample, p=ws)
                new_samples[i] = samples[idx] + np.dot(std, rng.randn(n_dim))
                dist = simulator(new_samples[i])
                n_sim += 1
                if verbose and n_sim % 1000 == 0: print("simulations:  ", n_sim,   "i:  ", i)
            # calculate unnormalized weights

            log_kernel = -0.5 * np.sum(sl.solve_triangular(std,
                                    (new_samples[i] - samples).T,
                                    lower=True) ** 2, axis=0)
            new_log_ws[i] = prior_logpdf(new_samples[i]) - logsumexp(log_ws + log_kernel)
        #print(new_log_ws.shape)
        #print(logsumexp)
        #print(logsumexp(new_log_ws))
        log_ws = new_log_ws - logsumexp(new_log_ws)
        samples = new_samples
        # Kish ESS
        log_ess = -logsumexp(2.0 * log_ws)# - log_n_sample
        if log_ess < log_ess_min:
            idx = rng.choice(n_sample, size=n_sample, p=np.exp(log_ws))
            samples = samples[idx]
            log_ws[:] = -log_n_sample
    return samples


def invgamma_solver(l, u):
    a = Symbol("a")
    b = ((u - l) / 6 *(a-1)**2 * (a-2)) ** 0.5
    res = solve(b/(a-1)+3*b**2/((a-1)**2 * (a-2)) - u, a)
    a = float(res[0])
    b = ((u - l) / 6 *(a-1)**2 * (a-2)) ** 0.5
    return (a, b)


