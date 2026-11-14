import numpy as np
from GPy.core.mapping import Mapping
from GPy.core.parameterization import Param

class QPoly(Mapping):
    """
    A sum of quadratic polynomials without cross-terms.
    .. math::
       F(\mathbf{x}) = \mathbf{A} \mathbf{x} ** 2 + \mathbf{B} \mathbf{x} + C
    :param input_dim: dimension of input.
    :type input_dim: int
    :param output_dim: dimension of output.
    :type output_dim: int
    """

    def __init__(self, input_dim, output_dim=1, A=None, B=None, C=None, seed=None,
                 name='qpolymap', set_zero=False):
        super(QPoly, self).__init__(input_dim=input_dim, output_dim=output_dim,
                                    name=name)
        random = np.random if seed is None else np.random.RandomState(seed)
        gen = (lambda x: random.normal(size=x)) if not set_zero else np.zeros
        if not A:
            A = gen((self.input_dim, self.output_dim))
        else:
            assert(A.shape == (input_dim, output_dim))
        if not B:
            B = gen((self.input_dim, self.output_dim))
        else:
            assert(A.shape == (input_dim, output_dim))
        if not C:
            C = 0 if set_zero else random.randn()
        else:
            if isinstance(C, np.ndarray):
                assert(C.shape == (output_dim, ))
            elif not isinstance(C, (int, float)):
                raise ValueError("C must be int, float or an np.ndarray.")
        C = np.atleast_1d(C)
        if C.size == 1:
            C = np.ones(self.output_dim) * C
        self.A = Param('A', A)
        self.link_parameter(self.A)
        self.B = Param('B', B)
        self.link_parameter(self.B)
        self.C = Param('C', C)
        self.link_parameter(self.C)

    def f(self, X): # X is a matrix, n_obs x output
        return np.dot(X ** 2, self.A) + np.dot(X, self.B) + \
               np.tile(self.C.values[None,:], (X.shape[0], 1))

    def update_gradients(self, dL_dF, X):
        self.A.gradient = np.dot((X ** 2).T, dL_dF)
        self.B.gradient = np.dot(X.T, dL_dF)
        self.C.gradient = dL_dF.sum(0)


    def gradients_X(self, dL_dF, X):
        return 2 * X * np.dot(dL_dF, self.A.T) + np.dot(dL_dF, self.B.T)

    def to_dict(self):
        input_dict = super(QPoly, self)._save_to_input_dict()
        input_dict["class"] = "GPy.mappings.QPoly"
        input_dict["A"] = self.A.values.tolist()
        input_dict["B"] = self.B.values.tolist()
        input_dict["C"] = self.C.values.tolist()
        return input_dict

    @staticmethod
    def _build_from_input_dict(mapping_class, input_dict):
        import copy
        input_dict = copy.deepcopy(input_dict)       
        l = QPoly(**input_dict)
        return l

class LogQPoly(Mapping):
    """
    Log of sum of quadratic polynomials without cross-terms.
    .. math::
       F(\mathbf{x}) = log(\mathbf{A} \mathbf{x} ** 2 + \mathbf{B} \mathbf{x} + C)
    :param input_dim: dimension of input.
    :type input_dim: int
    :param output_dim: dimension of output.
    :type output_dim: int
    """

    def __init__(self, input_dim, output_dim=1, A=None, B=None, C=None, seed=None,
                 name='logqpolymap'):
        super(LogQPoly, self).__init__(input_dim=input_dim, output_dim=output_dim,
                                    name=name)
        random = np.random if seed is None else np.random.RandomState(seed)
        if not A:
            A = random.randn(self.input_dim, self.output_dim)
        else:
            assert(A.shape == (input_dim, output_dim))
        if not B:
            B = random.randn(self.input_dim, self.output_dim)
        else:
            assert(B.shape == (input_dim, output_dim))
        if not C:
            C = 1.
        else:
            if isinstance(C, np.ndarray):
                assert(C.shape == (output_dim, ))
            elif not isinstance(C, (int, float)):
                raise ValueError("C must be int, float or an np.ndarray.")
        C = np.atleast_1d(C)
        if C.size == 1:
            C = np.ones(self.output_dim) * C
        self.A = Param('A', A)
        self.link_parameter(self.A)
        self.B = Param('B', B)
        self.link_parameter(self.B)
        self.C = Param('C', C)
        self.link_parameter(self.C)

    def f(self, X): # X is a matrix, n_obs x output
        return np.log(np.dot(X ** 2, self.A) + np.dot(X, self.B) + \
                      np.tile(self.C.values[None,:], (X.shape[0], 1)))

    def update_gradients(self, dL_dF, X):
        V = np.dot(X ** 2, self.A) + np.dot(X, self.B) + \
                      np.tile(self.C.values[None,:], (X.shape[0], 1))
        Vbar = (1 / V) * dL_dF
        self.A.gradient = np.dot((X ** 2).T, Vbar)
        self.B.gradient = np.dot(X.T, Vbar)
        self.C.gradient = Vbar.sum(0)


    def gradients_X(self, dL_dF, X):
        V = np.dot(X ** 2, self.A) + np.dot(X, self.B) + \
                      np.tile(self.C.values[None,:], (X.shape[0], 1))
        Vbar = (1 / V) * dL_dF
        return 2 * X * np.dot(Vbar, self.A.T) + np.dot(Vbar, self.B.T)

    def to_dict(self):
        input_dict = super(LogQPoly, self)._to_dict()
        input_dict["class"] = "GPy.mappings.LogQPoly"
        input_dict["A"] = self.A.values.tolist()
        input_dict["B"] = self.B.values.tolist()
        input_dict["C"] = self.C.values.tolist()
        return input_dict

class QuarticPoly(Mapping):
    """
    Quartic polynomials without cross-terms.
    .. math::
       F(\mathbf{x}) = \mathbf{A} \mathbf{x} ** 4 + \mathbf{B} \mathbf{x} **2 + \mathbf{C} \mathbf{x} + D
    :param input_dim: dimension of input.
    :type input_dim: int
    :param output_dim: dimension of output.
    :type output_dim: int
    """

    def __init__(self, input_dim, output_dim=1, A=None, B=None, C=None, D=None, seed=None,
                 name='quarticpolymap', set_zero=False):
        super(QuarticPoly, self).__init__(input_dim=input_dim, output_dim=output_dim,
                                    name=name)
        random = np.random if seed is None else np.random.RandomState(seed)
        gen = random.randn if not set_zero else np.zeros
        if not A:
            A = gen((self.input_dim, self.output_dim))
        else:
            assert(A.shape == (input_dim, output_dim))
        if not B:
            B = gen((self.input_dim, self.output_dim))
        else:
            assert(B.shape == (input_dim, output_dim))
        if not C:
            C = gen((self.input_dim, self.output_dim))
        else:
            assert(C.shape == (input_dim, output_dim))
        if not D:
            D = 0 if set_zero else random.randn()
        else:
            if isinstance(D, np.ndarray):
                assert(D.shape == (output_dim, ))
            elif not isinstance(D, (int, float)):
                raise ValueError("D must be int, float or an np.ndarray.")
        D = np.atleast_1d(D)
        if D.size == 1:
            D = np.ones(self.output_dim) * D
        self.A = Param('A', A)
        self.link_parameter(self.A)
        self.B = Param('B', B)
        self.link_parameter(self.B)
        self.C = Param('C', C)
        self.link_parameter(self.C)
        self.D = Param('D', D)
        self.link_parameter(self.D)

    def f(self, X): # X is a matrix, n_obs x output
        return np.dot(X ** 4, self.A) + np.dot(X ** 2, self.B) + np.dot(X, self.C) + \
               np.tile(self.D.values[None,:], (X.shape[0], 1))

    def update_gradients(self, dL_dF, X):
        self.A.gradient = np.dot((X ** 4).T, dL_dF)
        self.B.gradient = np.dot((X ** 2).T, dL_dF)
        self.C.gradient = np.dot(X.T, dL_dF)
        self.D.gradient = dL_dF.sum(0)

    def gradients_X(self, dL_dF, X):
        return 4 * X ** 3 * np.dot(dL_dF, self.A.T) +  2 * X * np.dot(dL_dF, self.B.T) + np.dot(dL_dF, self.C.T)

    def to_dict(self):
        input_dict = super(QuarticPoly, self)._to_dict()
        input_dict["class"] = "GPy.mappings.QuarticPoly"
        input_dict["A"] = self.A.values.tolist()
        input_dict["B"] = self.B.values.tolist()
        input_dict["C"] = self.C.values.tolist()
        input_dict["D"] = self.D.values.tolist()
        return input_dict

