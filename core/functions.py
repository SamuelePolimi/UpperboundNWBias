import numpy as np
import gym
from scipy.special import erf


class Orthopod:
    def __init__(self, d_n, d_p):
        """
        Constructs a orthopod all of dimension d.
        :param d_n: d array, the lower limit of the orthpods
        :param d_p: d array, the upper limit of the orthpods
        """
        assert hasattr(d_n, "shape") and hasattr(d_p, "shape"), "d_n and d_p should be ndarray"
        assert len(d_n.shape) == 1 and len(d_p.shape)==1, "d_n and d_p should be ndarray of dimension 1"
        assert d_n.shape[0] == d_p.shape[0], "d_n and d_p should have same dimensions"
        self.d_n = d_n
        self.d_p = d_p


class Orthopods:
    """
    This class represents one or many Orthopods
    """
    def __init__(self, d_n, d_p):
        """
        Constructs a generic array of n orthopods all of same dimension d.
        :param d_n: n x d array, the lower limit of the orthpods
        :param d_p: n x d array, the upper limit of the orthpods
        """
        assert hasattr(d_n, "shape") and hasattr(d_p, "shape"), "d_n and d_p should be ndarray"
        assert len(d_n.shape) == len(d_p.shape), "d_n and d_p should be ndarray of dimension 1"
        for i in range(len(d_n.shape)):
            assert d_n.shape[i] == d_p.shape[i], "d_n and d_p should have same dimensions"
        assert d_n.shape[1] == d_p.shape[1], "d_n and d_p should have same dimensions"
        self.d_n = d_n
        self.d_p = d_p


class LocalLipschitz:
    def __init__(self, X, L, domain):
        """
        This class represent the evaluation of Lipschitz condition on an array of points
        :type X: np.ndarray
        :type L: np.ndarray
        :type domain: Orthopods
        """
        self.X = X
        self.L = L
        self.domain = domain


class Function:

    def __call__(self, X, noise=True):
        pass

    def domain(self):
        """

        :return:
        :rtype: Orthopod
        """
        pass

    def first_derivate(self, X):
        pass

    def second_derivate(self, X):
        pass

    def local_lipschitz(self, X):
        """

        :param X:
        :return:
        :rtype: LocalLipschitz
        """
        pass

    def max_diffference(self):
        pass


class DensityFunction:

    def __call__(self, n_samples):
        pass

    def density(self, X):
        pass


    def domain(self):
        """

        :return:
        :rtype: Orthopod
        """
        pass

    def first_derivate(self, X):
        pass

    def second_derivate(self, X):
        pass

    def log_lipschitz(self, X):
        """

        :param X:
        :return:
        :rtype: LocalLipschitz
        """
        pass


class Sin(Function):

    def __init__(self, frequency, amplitude, phase):
        self.a = amplitude
        self.f = frequency
        self.b = phase

    def __call__(self, X, noise=True):
        return self.b + np.ravel(self.a * np.sin(self.f * X)) \
               + (np.random.normal(0., 0.05, size=X.shape[0]) if noise else 0.)

    def domain(self):
        u = np.array([1])
        return Orthopod(-u*np.infty, u*np.infty)

    def first_derivate(self, X):
        return np.ravel(self.a * self.f * np.cos(self.f * X))

    def second_derivate(self, X):
        return np.ravel(- self.a * self.f**2 * np.sin(self.f * X))

    def local_lipschitz(self, X):
        L = np.ones(X.shape[1:]) * self.a * self.f
        u = np.ones_like(X)
        domain = Orthopods(-u * np.infty, u * np.infty)
        return LocalLipschitz(X, L, domain)

    def max_diffference(self):
        return 2 * self.a


class Pendulum(Function):

    def __init__(self):
        self.env = gym.make("Pendulum-v0")
        self.u = np.array([np.pi, 8., 2.])

    def __call__(self, X, noise=True):
        self.env.reset()
        ret = []
        for x in X:
            self.env.state = x[:2]
            self.env.step(x[-1:])
            ret.append(self.env.state[0])
        return np.array(ret)

    def domain(self):
        return Orthopod(-self.u, self.u)

    def first_derivate(self, X):
        raise NotImplemented()

    def second_derivate(self, X):
        raise NotImplemented()

    def local_lipschitz(self, X):
        L = np.ones(X.shape[1:])        # Appears that the lipschitz function is 1
        U = np.repeat(self.u[:, np.newaxis], X.shape[1], axis=1)

        domain = Orthopods(-U, U)

        return LocalLipschitz(X, L, domain)

    def max_diffference(self):
        return 2 * np.pi


class Lin(Function):

    def __init__(self, d):
        self.d = d

    def __call__(self, X, noise=True):
        return np.sum(X, axis=1) + (np.random.normal(0., 0.005, size=X.shape[0]) if noise else 0.)

    def domain(self):
        u = np.ones(self.d)
        return Orthopod(-u*np.infty, u*np.infty)

    def first_derivate(self, X):
        raise Exception("Only used in Rosenblatt")

    def second_derivate(self, X):
        raise Exception("Only used in Rosenblatt")

    def local_lipschitz(self, X):
        L = np.ones(X.shape[1:])
        u = np.ones_like(X)
        domain = Orthopods(-u * np.infty, u * np.infty)
        return LocalLipschitz(X, L, domain)

    def max_diffference(self):
        return np.inf


class LnCosh(Function):

    def __init__(self, a, c):
       self.a = a
       self.c = c

    def __call__(self, X, noise=True):
        return np.ravel(1. / self.a * np.log(np.cosh(self.a * (X + self.c)))) \
              + (np.random.normal(0., 0.05, size=X.shape[0]) if noise else 0.)

    def domain(self):
        u = np.array([1])
        return Orthopod(-u*np.infty, u*np.infty)

    def first_derivate(self, X):
        return np.ravel(np.tanh(self.a * (X + self.c)))

    def second_derivate(self, X):
        return np.ravel(self.a / np.cosh(self.a * (X + self.c)) ** 2)

    def local_lipschitz(self, X):
        L = np.ones(X.shape[1:])
        u = np.ones_like(X)
        domain = Orthopods(u * - np.infty, u * np.infty)
        return LocalLipschitz(X, L, domain)

    def max_diffference(self):
        return np.infty


class Root(Function):

    def __init__(self, scale):
        self.s = scale

    def __call__(self, X, noise=True):
        return np.ravel(np.sqrt(np.square(self.s * X) + 1)) \
               + (np.random.normal(0., 0.05, size=X.shape[0]) if noise else 0.)

    def domain(self):
        u = np.array([1])
        return Orthopod(-u*np.infty, u*np.infty)

    def first_derivate(self, X):
        return np.ravel(self.s * X) / self(X, noise=False)

    def second_derivate(self, X):
        return np.ravel(self.s / self(X, noise=False)) - np.ravel(self.s**2*X**2)/self(X, noise=False)**3

    def local_lipschitz(self, X):
        L = np.ones(X.shape[1:]) * self.s
        u = np.ones_like(X)
        domain = Orthopods(u * - np.infty, u * np.infty)
        return LocalLipschitz(X, L, domain)

    def max_diffference(self):
        return np.infty


class Log(Function):

    def __init__(self):
        pass

    def __call__(self, X, noise=True):
        return np.ravel(np.log(X))

    def domain(self):
        u = np.array([1])
        return Orthopod(-u*0., u*np.infty)

    def first_derivate(self, X):
        return 1. / np.ravel(X)

    def second_derivate(self, X):
        return - 1 /  np.ravel(X)**2

    def local_lipschitz(self, X):
        """
        Log is lipschitz between 0 and +infty
        :param X:
        :return:
        """
        L = np.ones(X.shape[1:])
        u = np.ones_like(X)
        domain = Orthopods(u * 0, u * np.infty)
        return LocalLipschitz(X, L, domain)

    def max_diffference(self):
        return np.infty


class Pareto(DensityFunction):

        def __init__(self, c=1., alpha=1.):
            self.c = c
            self.alpha = alpha

        def __call__(self, n_samples):
            return np.random.pareto(self.alpha, size=n_samples).reshape(-1, 1) + 1.

        def domain(self):
            u = np.array([1])
            return Orthopod(u, u * np.infty)

        def density(self, X):
            ret = self.alpha / np.power(np.abs(np.ravel(X)), self.alpha + 1.)
            ret[np.ravel(X) < 1.] = 0.
            return ret

        def first_derivate(self, X):
            return - self.alpha * (self.alpha + 1.) / np.power(np.abs(np.ravel(X)), self.alpha + 2.)

        def log_lipschitz(self, X):
            L = np.ones(X.shape[1:]) * (self.alpha + 1)
            u = np.ones_like(X)
            domain = Orthopods(u * 1., u * np.infty)
            return LocalLipschitz(X, L, domain)


class Cauchy(DensityFunction):

    def __init__(self, mu=0., gamma=1.):
        self.mu = mu
        self.gamma = gamma

    def __call__(self, n_samples):
        # TODO: double check
        return self.gamma * np.random.standard_cauchy(n_samples).reshape(-1, 1) + self.mu - self.gamma

    def domain(self):
         u = np.array([1])
         return Orthopod(-u * np.infty, u * np.infty)

    def density(self, X):
        # TODO: checkx = np!
        return np.ravel(1. / (np.pi * self.gamma * (1 + ((X-self.mu)/self.gamma)**2)))

    def first_derivate(self, X):
        return - np.ravel(X - self.mu) / np.ravel(np.pi * self.gamma**3 * (1 + ((X-self.mu)/self.gamma)**2)**2)

    def derivate_log_density(self, x):
        return - 2 * (x - self.mu)/(self.gamma**2 * (1 + (x - self.mu)**2/self.gamma**2))

    def log_lipschitz(self, X):
        max_x = 2*self.mu + self.gamma**2 + np.sqrt(np.abs(4*self.mu**2 + self.gamma**4 + 2*self.mu*self.gamma**2
                                                           - 4*(self.gamma**2 + self.mu + self.gamma**2*self.mu)))
        max_x = max_x/2.
        L = np.ones(X.shape[1:]) * np.abs(self.derivate_log_density(max_x))
        u = np.ones_like(X)
        domain = Orthopods(-u * np.infty, u * np.infty)
        return LocalLipschitz(X, L, domain)


class Laplace(DensityFunction):

    def __init__(self, mean=0., laplace_c=0.5):
        self.l = laplace_c
        self.m = mean

    def __call__(self, n_samples):
        return np.random.laplace(self.m, self.l, n_samples).reshape(-1, 1)

    def domain(self):
        u = np.array([1])
        return Orthopod(-u * np.infty, u * np.infty)

    def density(self, X):
        return np.ravel(1. / (2 * self.l) * np.exp(-np.abs(X - self.m) / self.l))

    def first_derivate(self, X):
        return - np.ravel(np.sign(X - self.m) * 1./(2*self.l**2) * np.exp(-np.abs(X- self.m) / self.l))

    def log_lipschitz(self, X):
        L = np.ones(X.shape[1:]) * 1./self.l
        u = np.ones_like(X)
        domain = Orthopods(-u * np.infty, u * np.infty)
        return LocalLipschitz(X, L, domain)


class Uniform(DensityFunction):

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, n_samples):
        return np.random.uniform(self.a, self.b, size=n_samples).reshape(-1, 1)

    def domain(self):
        u = np.array([1])
        return Orthopod(u * self.a, u * self.b)

    def density(self, X):
        d = 1/(self.b-self.a)
        Y = np.ravel(np.where(np.logical_and(self.a <= X, X <= self.b), d, 0.))
        return np.ravel(Y)

    def first_derivate(self, X):
        return np.zeros(X.shape[0])

    def log_lipschitz(self, X):
        L = np.zeros(X.shape[1:])           # To make computation feasible also for other kernels
        u = np.ones_like(X)
        domain = Orthopods(u * self.a, u * self.b)
        return LocalLipschitz(X, L, domain)


class MultiUniform(DensityFunction):

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.d = a.shape[0]

    def __call__(self, n_samples):
        return np.random.uniform(self.a, self.b, size=(n_samples, self.d))

    def domain(self):
        return Orthopod(self.a, self.b)

    def density(self, X):
        volume = 1/np.prod(self.b-self.a)
        # For just this experiment we don't need to check
        #Y = np.ravel(np.where(np.logical_and(self.a <= X, X <= self.b), volume, 0.))
        return np.ones(X.shape[0]) * volume

    def first_derivate(self, X):
        return np.zeros(X.shape[0])

    def log_lipschitz(self, X):
        L = np.zeros(X.shape[1:])
        # u = np.ones_like(X)
        domain = Orthopods(np.array([self.a]*X.shape[1]).T, np.array([self.b]*X.shape[1]).T)
        return LocalLipschitz(X, L, domain)


class MultiGaussian(DensityFunction):

    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma
        self.d = mean.shape[0]
        self.a = - np.ones(self.d)*np.infty
        self.b =  np.ones(self.d)*np.infty

    def __call__(self, n_samples):
        return np.random.multivariate_normal(self.mean, np.diag(self.sigma**2), size=n_samples)

    def domain(self):
        return Orthopod(self.b, self.a)

    def density(self, X):
        # volume = 1/np.prod(self.b-self.a)
        # For just this experiment we don't need to check
        #Y = np.ravel(np.where(np.logical_and(self.a <= X, X <= self.b), volume, 0.))
        #return np.ones(X.shape[0]) * volume
        raise NotImplemented()

    def first_derivate(self, X):
        raise NotImplemented()

    def log_lipschitz(self, X, delta=np.infty):

        lipschitz = None
        for i in range(X.shape[0]):
            x = np.ravel(X[i])
            u = np.ones_like(x)
            x_max = u * self.sigma[i]
            x_min = -u * self.sigma[i]

            x_max[x_max < x] = 0.
            x_min[x_min > x] = 0.

            assert X is not None, "For Gaussian Distribution we need a Local definition of Lipschitz continuity."
            f = lambda x: -0.5 * np.square(x - self.mean[i]) / self.sigma[i]**2

            p = (f(x_max) - f(x))/(x_max - x)
            n = (f(x_min) - f(x))/delta
            if lipschitz is None:
                lipschitz = np.max([np.abs(p), np.abs(n)], axis=0)
            else:
                lipschitz = np.max([np.abs(p), np.abs(n), lipschitz], axis=0)

        domain = Orthopods(np.array([self.a] * X.shape[1]).T, np.array([self.b] * X.shape[1]).T)
        return LocalLipschitz(X, lipschitz, domain)


#TODO: elaborate
class Normal(DensityFunction):

    def __init__(self, mean=0., sigma=0.5):
        self.sigma = sigma
        self.m = mean

    def __call__(self, n_samples):
        return np.random.normal(self.m, self.sigma, n_samples).reshape(-1, 1)

    def density(self, X):
        return np.ravel(1./np.sqrt(2*np.pi*self.sigma**2) * np.exp(-0.5 * np.square(X- self.m) / self.sigma**2))

    def first_derivate(self, X):
        return -np.ravel(1./np.sqrt(2*np.pi*self.sigma**6) *
                         (X - self.m) * np.exp(-0.5 * np.square(X - self.m) / self.sigma**2))

    def log_lipschitz(self, X=None, delta=np.infty):
        """
        Let's take the local lipschitz constant at 2sigma
        :return:
        """
        x = np.ravel(X)
        u = np.ones_like(x)
        x_max = u * self.sigma
        x_min = -u * self.sigma

        x_max[x_max < x] = 0.
        x_min[x_min > x] = 0.

        assert X is not None, "For Gaussian Distribution we need a Local definition of Lipschitz continuity."
        f = lambda x: -0.5 * np.square(x - self.m) / self.sigma**2

        p = np.zeros_like(x)
        n = np.zeros_like(x)
        p = (f(x_max) - f(X))/(x_max - x)
        n = (f(x_min) - f(X))/delta
        return np.max([np.abs(p), np.abs(n)], axis=0)