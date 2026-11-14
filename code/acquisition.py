import numpy as np
import scipy.stats as ss
from utils import minimize
from qpoly import QPoly, LogQPoly, QuarticPoly

class AcquisitionBase:
    """All acquisition functions are assumed to fulfill this interface.
    Gaussian noise ~ N(0, self.noise_var) is added to the acquired points. By default,
    noise_var=0. You can define a different variance for the separate dimensions.
    """

    def __init__(self,
                 model,
                 bounds,
                 prior=None,
                 n_inits=10,
                 max_opt_iters=1000,
                 noise_var=0,
                 exploration_rate=10,
                 seed=None):
        """Initialize AcquisitionBase.
        Parameters
        ----------
        model : GPy.model.GPRegression instance
            Gaussian Process Regression model.
        bounds : tuple of length 'model.input_dim' of tuples (min, max)
            The region where to estimate the posterior for each parameter.
        prior : scipy-like distribution, optional
            By default uniform distribution within model bounds.
        n_inits : int, optional
            Number of initialization points in internal optimization.
        max_opt_iters : int, optional
            Max iterations to optimize when finding the next point.
        noise_var : float or np.array, optional
            Acquisition noise variance for adding noise to the points near the optimized
            location. If array, must be 1d specifying the variance for different dimensions.
            Default: no added noise.
        exploration_rate : float, optional
            Exploration rate of the acquisition function (if supported)
        seed : int, optional
            Seed for getting consistent acquisition results. Used in getting random
            starting locations in acquisition function optimization.
        """
        self.model = model
        self.bounds = bounds # added
        self.prior = prior
        self.n_inits = int(n_inits)
        self.max_opt_iters = int(max_opt_iters)
        if isinstance(noise_var, float):
            noise_var = np.full(model.input_dim, noise_var)
        assert(noise_var.size == model.input_dim)
        self.noise_var = noise_var
        self.exploration_rate = exploration_rate
        self.random_state = np.random if seed is None else np.random.RandomState(seed)
        self.seed = 0 if seed is None else seed

    def evaluate(self, x, t=None):
        """Evaluate the acquisition function at 'x'.
        Parameters
        ----------
        x : numpy.array
        t : int
            current iteration (starting from 0)
        """
        raise NotImplementedError

    def evaluate_gradient(self, x, t=None):
        """Evaluate the gradient of acquisition function at 'x'.
        Parameters
        ----------
        x : numpy.array
        t : int
            Current iteration (starting from 0).
        """
        raise NotImplementedError

    def acquire(self, n, t=None, return_nonoise=False):
        """Return the next batch of acquisition points.
        Gaussian noise ~ N(0, self.noise_var) is added to the acquired points.
        Parameters
        ----------
        n : int
            Number of acquisition points to return.
        t : int
            Current acq_batch_index (starting from 0).
        Returns
        -------
        x : np.ndarray
            The shape is (n, input_dim)
        """
        # Optimize the current minimum
        xhat, _ = minimize(
            lambda x: self.evaluate(x, t),
            self.bounds,
            lambda x: self.evaluate_gradient(x, t),
            self.prior,
            self.n_inits,
            self.max_opt_iters,
            random_state=self.random_state)
        # Create n copies of the minimum
        x = np.tile(xhat, (n, 1))
        if np.any(self.noise_var != 0):
            x = self._add_noise(x)
        return x if not return_nonoise else (x, xhat)

    def _add_noise(self, x):
        for i in range(self.model.input_dim):
            std = np.sqrt(self.noise_var[i])
            if std == 0:
                continue
            xi = x[:, i]
            a = (self.bounds[i][0] - xi) / std
            b = (self.bounds[i][1] - xi) / std
            x[:, i] = ss.truncnorm.rvs(
                a, b, loc=xi, scale=std, size=len(x), random_state=self.random_state)
        return x

class LCBSC(AcquisitionBase):
    r"""Lower Confidence Bound Selection Criterion.
    Srinivas et al. call this GP-LCB.
    LCBSC uses the parameter delta which is here equivalent to 1/exploration_rate.
    Parameter delta should be in (0, 1) for the theoretical results to hold. The
    theoretical upper bound for total regret in Srinivas et al. has a probability greater
    or equal to 1 - delta, so values of delta very close to 1 or over it do not make much
    sense in that respect.
    Delta is roughly the exploitation tendency of the acquisition function.
    References
    ----------
    N. Srinivas, A. Krause, S. M. Kakade, and M. Seeger. Gaussian
    process optimization in the bandit setting: No regret and experimental design. In
    Proc. International Conference on Machine Learning (ICML), 2010
    E. Brochu, V.M. Cora, and N. de Freitas. A tutorial on Bayesian optimization of expensive
    cost functions, with application to active user modeling and hierarchical reinforcement
    learning. arXiv:1012.2599, 2010.
    Notes
    -----
    The formula presented in Brochu (pp. 15) seems to be from Srinivas et al. Theorem 2.
    However, instead of having t**(d/2 + 2) in \beta_t, it seems that the correct form
    would be t**(2d + 2).
    """

    def __init__(self, *args, delta=None, **kwargs):
        """Initialize LCBSC.
        Parameters
        ----------
        args
        delta : float, optional
            In between (0, 1). Default is 1/exploration_rate. If given, overrides the
            exploration_rate.
        kwargs
        """
        if delta is not None:
            kwargs['exploration_rate'] = 1 / delta
        super(LCBSC, self).__init__(*args, **kwargs)
        self.name = 'lcbsc'
        self.label_fn = 'Confidence Bound'

    @property
    def delta(self):
        """Return the inverse of exploration rate."""
        return 1 / self.exploration_rate

    def set_model(self, model):
        self.model = model

    def _beta(self, t):
        # Start from 0
        t += 1
        d = self.model.input_dim
        return 2 * np.log(t**(2 * d + 2) * np.pi**2 / (3 * self.delta))

    def evaluate(self, x, t=None, kern=None):
        """Evaluate the Lower confidence bound selection criterion.
        mean - sqrt(\beta_t) * std
        Parameters
        ----------
        x : numpy.array
        t : int
            Current iteration (starting from 0).
        """
        if kern is not None and x.ndim == 1:
            z = np.zeros(self.model.input_dim)
            z[kern.active_dims] = x
            x = z[None,:]
        x = np.atleast_2d(x)
        mean, var = self.model.predict_noiseless(x, kern=kern)
        return (mean - np.sqrt(self._beta(t) * var))[0][0]

    def evaluate_gradient(self, x, t=None, kern=None):
        """Evaluate the gradient of the lower confidence bound selection criterion.
        Parameters
        ----------
        x : numpy.array
        t : int
            Current iteration (starting from 0).
        """
        if kern is not None and x.ndim == 1:
            z = np.zeros(self.model.input_dim)
            z[kern.active_dims] = x
            x = z[None,:]
        x = np.atleast_2d(x)
        mean, var = self.model.predict_noiseless(x, kern=kern)
        grad_mean, grad_var = self.model.predictive_gradients(x, kern=kern)
        grad_mean = grad_mean[:,:,0] if kern is None else grad_mean[:, kern.active_dims, 0]
        grad_var = grad_var if kern is None else grad_var[:, kern.active_dims]
        return (grad_mean - 0.5 * grad_var * np.sqrt(self._beta(t) / var))[0]

class LCBSCADD(LCBSC):
    def beta(self, t, d):
        # Start from 0
        t += 1
        return  0.2*d*np.log(2*t)

    def evaluate(self, x, d, t=None, kern=None):
        """Evaluate the Lower confidence bound selection criterion.
        mean - sqrt(\beta_t) * std
        Parameters
        ----------
        x : numpy.array
        t : int
            Current iteration (starting from 0).
        """
        if kern is not None and x.ndim == 1:
            z = np.zeros(self.model.input_dim)
            z[kern.active_dims] = x
            x = z[None,:]
        x = np.atleast_2d(x)
        mean, var = self.model.predict_noiseless(x, kern=kern)
        return (mean - np.sqrt(self.beta(t, d) * var))[0][0]

    def evaluate_gradient(self, x, d, t=None, kern=None):
        """Evaluate the gradient of the lower confidence bound selection criterion.
        Parameters
        ----------
        x : numpy.array
        t : int
            Current iteration (starting from 0).
        """
        if kern is not None and x.ndim == 1:
            z = np.zeros(self.model.input_dim)
            z[kern.active_dims] = x
            x = z[None,:]
        x = np.atleast_2d(x)
        mean, var = self.model.predict_noiseless(x, kern=kern)
        grad_mean, grad_var = self.model.predictive_gradients(x, kern=kern)
        grad_mean = grad_mean[:,:,0] if kern is None else grad_mean[:, kern.active_dims, 0]
        grad_var = grad_var if kern is None else grad_var[:, kern.active_dims]
        return (grad_mean - 0.5 * grad_var * np.sqrt(self.beta(t, d) / var))[0]

    def acquire(self, n, t=None, return_nonoise=False):
        """Return the next batch of acquisition points.
        Gaussian noise ~ N(0, self.noise_var) is added to the acquired points.
        Parameters
        ----------
        n : int
            Number of acquisition points to return.
        t : int
            Current acq_batch_index (starting from 0).
        Returns
        -------
        x : np.ndarray
            The shape is (n, input_dim)
        """
        # Optimize the current minimum (DOES NOT WORK WITH PRIOR, must be iterable)
        d = np.max([len(kernel.active_dims) for kernel in self.model.kern.parts])
        xhat = np.zeros(self.model.input_dim)
        if isinstance(self.model.mean_function, (QPoly, LogQPoly, QuarticPoly)):
            orig = self.model.mean_function.copy()
            nparts = float(len(self.model.kern.parts))
        for kernel in self.model.kern.parts:
            active_dims = kernel.active_dims
            if isinstance(self.model.mean_function, (QPoly, LogQPoly, QuarticPoly)):
                mask = np.zeros(self.model.input_dim, dtype=bool)
                mask[active_dims] = True
                self.model.mean_function = orig.copy()
                self.model.mean_function.A = self.model.mean_function.A.values * mask[:,None]
                self.model.mean_function.B = self.model.mean_function.B.values * mask[:,None]
                if isinstance(self.model.mean_function, QuarticPoly):
                    self.model.mean_function.C = self.model.mean_function.C.values * mask[:,None]
                    self.model.mean_function.D = self.model.mean_function.D.values / nparts
                else:
                    self.model.mean_function.C = self.model.mean_function.C.values / nparts
            xhati, _ = minimize(
                lambda x: self.evaluate(x, d, t, kern=kernel),
                self.bounds[active_dims],
                lambda x: self.evaluate_gradient(x, d, t, kern=kernel),
                self.prior,
                self.n_inits,
                self.max_opt_iters,
                random_state=self.random_state)
            xhat[active_dims] = xhati
        if isinstance(self.model.mean_function, (QPoly, LogQPoly, QuarticPoly)):
            self.model.mean_function = orig

        # Create n copies of the minimum
        x = np.tile(xhat, (n, 1))
        if np.any(self.noise_var != 0):
            x = self._add_noise(x)
        return x if not return_nonoise else (x, xhat)