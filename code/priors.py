import numpy as np
from scipy.special import gammaln
from paramz.domains import _REAL, _POSITIVE, _NEGATIVE
import GPy
import weakref

class InverseGamma(GPy.priors.Gamma):
    """
    Implementation of the inverse-Gamma probability function, coupled with random variables.

    :param a: shape parameter
    :param b: rate parameter (warning: it's the *inverse* of the scale)

    .. Note:: Bishop 2006 notation is used throughout the code

    """
    domain = _POSITIVE
    _instances = []
    def __new__(cls, a=1, b=.5): # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().a == a and instance().b == b:
                    return instance()
        #o = super(Prior, cls).__new__(cls, a, b) # BUGGED
        #cls._instances.append(weakref.ref(o))
        #return cls._instances[-1]()
        newfunc = super(GPy.priors.Prior, cls).__new__ 
        if newfunc is object.__new__:
            o = newfunc(cls)  
        else:
            o = newfunc(cls, a, b)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, a, b):
        self._a = float(a)
        self._b = float(b)
        self.constant = -gammaln(self.a) + a * np.log(b)

    def __str__(self):
        return "iGa({:.2g}, {:.2g})".format(self.a, self.b)

    def lnpdf(self, x):
        x = x + 1e-20
        return self.constant - (self.a + 1) * np.log(x) - self.b / x

    def lnpdf_grad(self, x):
        x = x + 1e-20
        return -(self.a + 1.) / x + self.b / x ** 2

    def rvs(self, n):
        return 1. / np.random.gamma(scale=1. / self.b, shape=self.a, size=n)

class HalfT(GPy.priors.Prior):
    """
    Implementation of the half student t probability function, coupled with random variables.

    :param A: scale parameter
    :param nu: degrees of freedom

    """
    domain = _POSITIVE
    _instances = []

    def __new__(cls, A, nu):  # Singleton: # THIS WAS BUGGED
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().A == A and instance().nu == nu:
                    return instance()
        #o = super(GPy.priors.Prior, cls).__new__(cls, A, nu) 
        #cls._instances.append(weakref.ref(o))
        #return cls._instances[-1]()
        newfunc = super(GPy.priors.Prior, cls).__new__ 
        if newfunc is object.__new__:
            o = newfunc(cls)  
        else:
            o = newfunc(cls, A, nu)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, A, nu):
        self.A = float(A)
        self.nu = float(nu)
        self.constant = np.log(2) + gammaln(.5*(self.nu+1.)) - gammaln(.5*self.nu) - .5*np.log(np.pi*self.A**2*self.nu) # HAD BUG

    def __str__(self):
        return "hT({:.2g}, {:.2g})".format(self.A, self.nu)

    def lnpdf(self, theta):
        return (theta > 0) * (self.constant - .5*(self.nu + 1) * np.log(1. + (1./self.nu) * (theta/self.A)**2))

    def lnpdf_grad(self, theta):
        theta = theta if isinstance(theta, np.ndarray) else np.array([theta])
        grad = np.zeros_like(theta)
        above_zero = theta > 1e-6
        v = self.nu
        sigma2 = self.A**2 # HAD BUG
        grad[above_zero] = -0.5*(v+1)*(2*theta[above_zero])/(v*sigma2 + theta[above_zero][0]**2)
        return grad


class Exponential(GPy.priors.Prior):
    """
    Implementation of the Exponential probability function,
    coupled with random variables.

    :param l: shape parameter

    """
    domain = _POSITIVE
    _instances = []

    def __new__(cls, l):  # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if instance().l == l:
                    return instance()
        #o = super(Exponential, cls).__new__(cls, l) # THIS HAD A BUG
        newfunc = super(Exponential, cls).__new__
        if newfunc is object.__new__:
            o = newfunc(cls)  
        else:
            o = newfunc(cls, l)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, l):
        self.l = l

    def __str__(self):
        return "Exp({:.2g})".format(self.l)

    def summary(self):
        ret = {"E[x]": 1. / self.l,
               "E[ln x]": np.nan,
               "var[x]": 1. / self.l**2,
               "Entropy": 1. - np.log(self.l),
               "Mode": 0.}
        return ret

    def lnpdf(self, x):
        return np.log(self.l) - self.l * x

    def lnpdf_grad(self, x):
        return - self.l

    def rvs(self, n):
        return np.random.exponential(scale=1/self.l, size=n) # THIS HAD A BUG its supposed to be 1/l
