import numpy as np
from scipy.special import erf

otherwise = lambda x: np.full_like(x, True, dtype=bool)


def constant(c):
    return lambda x: np.ones_like(x) * c


def smaller_of(val1):
    return lambda x: x <= val1


def bigger_of(val1):
    return lambda x: x >= val1


def between(val1, val2):
    return lambda x: np.logical_and(val1 <= x, x <= val2)


class ConditionalFunction:

    def __init__(self, *cases):
        self._values = []
        self._conditions = []
        for case in cases:
            self._values.append(case[0])
            self._conditions.append(case[1])

    def __call__(self, x, *args):
        ret = np.ones_like(x) * np.nan
        for value, condition in zip(self._values[::-1], self._conditions[::-1]):
            idx = condition(x)
            x_true = x[idx]
            ret[idx] = value(x_true, *[arg[idx] for arg in args])
        return ret


class Kernel:

    def get_dims(self):
        pass

    def get_bandwidth(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def get_phi(self, a):
        pass

    def get_b(self, a, h):
        pass

    def get_c(self, a, h):
        pass

    def get_rosenblatt_constant(self):
        pass


def get_zeta(kernel, L_f, h, a, b):
    ret = h*(kernel.get_b(b/h, -h*L_f) - kernel.get_b(-a/h, h*L_f))
    ret[a < 0] = np.nan
    ret[b < 0] = np.nan
    return ret


def get_psi(kernel, L_f, h, a, b):
    ret = h*(kernel.get_b(b/h, h * L_f) - kernel.get_b(-a/h, h * L_f))
    ret[a < 0] = np.nan
    ret[b < 0] = np.nan
    return ret


def get_xi(kernel, L_f, h, a, b):
    ret = h**2*(kernel.get_c(b/h, h * L_f) + kernel.get_c(-a/h, h * L_f))
    ret[a < 0] = np.nan
    ret[b < 0] = np.nan
    return ret


def get_D(kernel, d, L_f, h, a, b):
    ret = 2**d * np.prod(h, axis=0) * np.prod(kernel.get_phi(np.infty/h), axis=0) - np.prod(h, axis=0) * np.prod(kernel.get_phi(b/h) + kernel.get_phi(a/h), axis=0)
    # ret[a > 0] = np.nan
    # ret[b < 0] = np.nan
    return ret


class GaussianKernel(Kernel):

    def __init__(self, bandwidths):
        self.h = bandwidths
        self.k = bandwidths.shape[0]
        self.precision = np.diag(1./self.h**2)

    def __call__(self, delta):
        """
        Returns the kernel for n samples
        :param delta: thus array must have the last shape=d
        :return:
        """
        n_dim = len(delta.shape)
        _delta = np.expand_dims(delta, axis=n_dim)
        # mahalanobis = np.swapaxes(_delta, n_dim-1, n_dim) @ self.precision @ _delta/2.
        #
        # return np.squeeze(np.exp(-mahalanobis)/np.sqrt((2. * np.pi)**self.k / np.prod(self.h**2)), axis=(n_dim-1, n_dim))

        mahalanobis = np.swapaxes(_delta, n_dim - 1, n_dim) @ self.precision @ _delta

        return np.squeeze(np.exp(-mahalanobis),
                          axis=(n_dim - 1, n_dim))

    def get_phi(self, a):
        return np.sqrt(np.pi)/2 * erf(a)

    def get_b(self, a, L):
        return np.sqrt(np.pi)/2. * np.exp(L**2/4)*(erf(a + L/2) - erf(L/2))

    def get_c(self, a, L):
        ret = np.exp(-a*(a+L))
        ret[a == np.infty] = 0.
        ret[a == -np.infty] = 0.
        return 0.5*(1 - ret - L*self.get_b(a, L))

    def get_dims(self):
        return self.k

    def get_bandwidth(self):
        return self.h

    def get_rosenblatt_constant(self):
        return 1/2    # (sqrt(pi)/2; sqrt(pi))


class Box(Kernel):

    def __init__(self, bandwidths):
        self.h = bandwidths
        self.k = bandwidths.shape[0]
        self.f = ConditionalFunction((lambda x: 0., lambda x: np.logical_or(x < -1, x > 1)),
                        (lambda x: 1., lambda x: np.logical_and(-1 <= x, x <= 1.)))

    def __call__(self, delta):
        """
        Returns the kernel for n samples
        :param delta: thus array must have the last shape=d
        :return:
        """
        res = []
        for i, h in enumerate(self.h):
            x = delta[..., i]/h
            res.append(self.f(x/h))
        return np.prod(res, axis=0)

    def get_dims(self):
        return self.k

    def get_bandwidth(self):
        return self.h

    def get_phi(self, a):
        u = np.copy(a)
        u[a < -1] = 0.
        u[a > 1] = 1.
        return u

    def get_b(self, a, L_i):
        f = ConditionalFunction(
            (lambda x, L: np.zeros_like(x), smaller_of(-1)),
            (lambda x, L: np.exp(-L)*(np.exp(2*L)-1)/L, bigger_of(1)),
            (lambda x, L: (np.exp(L)-np.exp(-L*x))/L, otherwise))
        return f(a, L_i) - f(np.zeros_like(a), L_i)

    def get_c(self, a, L_i):
        f = ConditionalFunction((lambda x, L: np.zeros_like(x), smaller_of(-1)),
                                (lambda x, L: np.exp(-L)*(-np.exp(2*L)*(L-1)-L-1)/L**2, bigger_of(1)),
                                (lambda x, L: -(np.exp(-L*x)*(L*x+1) + np.exp(L)*(L-1))/L**2, otherwise))
        return f(a, L_i) - f(np.zeros_like(a), L_i)

    def get_rosenblatt_constant(self):
        return 1/3


class Triangle(Kernel):

    def __init__(self, bandwidths):
        self.h = bandwidths
        self.k = bandwidths.shape[0]
        self.f = ConditionalFunction((lambda x: 1 - np.abs(x), between(-1, 1)),
                                     (lambda x: np.zeros_like(x), otherwise))

    def __call__(self, delta):
        """
        Returns the kernel for n samples
        :param delta: thus array must have the last shape=d
        :return:
        """
        res = []
        for i, h in enumerate(self.h):
            x = delta[..., i]/h
            res.append(self.f(x/h))
        return np.prod(res, axis=0)

    def get_dims(self):
        return self.k

    def get_bandwidth(self):
        return self.h

    def get_phi(self, a):
        f = ConditionalFunction((constant(0.), bigger_of(1)),
                                (lambda x: (x + 1)**2/2, between(-1, 0)),
                                (lambda x: x - x**2/2 + 0.5, between(0, 1)),
                                (constant(1), otherwise))
        return f(a)-f(np.zeros_like(a))

    def get_b(self, a, L_i):
        f = ConditionalFunction((lambda x, L: np.exp(-L)*(np.exp(L)-1)**2/L**2, bigger_of(1)),
                                (lambda x, L: np.exp(-L*x)*(-L*(x+1) + np.exp(L*x + L) - 1)/L**2, between(-1, 0)),
                                (lambda x, L: np.exp(-L*x)*(L*(x-1) - 2*np.exp(L*x) + np.exp(L*x + L) + 1)/L**2, between(0, 1)),
                                (lambda x, L: np.zeros_like(x), otherwise))
        return f(a, L_i)-f(np.zeros_like(a), L_i)

    def get_c(self, a, L_i):
        f = ConditionalFunction((lambda x, L: -np.exp(-L)*(np.exp(L)-1)*(np.exp(L)*(L-2) + L + 2)/L**3, bigger_of(1)),
                                (lambda x, L: np.exp(-L*x)*(L**2*(x-1)*x + L*(2*x - 1) - 4*np.exp(L*x) - (L-2)*np.exp(L*x + L) + 2)/L**3, between(0, 1)),
                                (lambda x, L: np.exp(-L*x)*(L**2*(x+1)*(-x) - L*(2*x + 1) - (L-2)*np.exp(L*x + L) - 2)/L**3, between(1, 0)),
                                (lambda x, L: np.zeros_like(x), otherwise))
        return f(a, L_i)-f(np.zeros_like(a), L_i)

    def get_rosenblatt_constant(self):
        return 1/6
