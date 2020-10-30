import numpy as np
from scipy.special import erf

from .kernels import Kernel
from .functions import DensityFunction, Function


def phi(tau, L, h):
        ret = np.exp(-0.5 * tau**2/h**2 - tau* L)
        ret[tau == -np.infty] = 0.
        ret[tau == np.infty] = 0.
        return ret


def varphi(tau, L, h):
        ret = erf((tau + h**2*L)/(h*np.sqrt(2)))
        ret[tau == np.infty] = 1.
        ret[tau == -np.infty] = -1.
        return ret


def Psi(L, h, tau_n, tau_p):
        return np.exp(L**2 * h**2/2.) * (varphi(tau_p, L, h) - varphi(tau_n, L, h))


def Zeta(L, h, tau_n, tau_p):
        return np.exp(L ** 2 * h ** 2 / 2.) *(
                2 * varphi(0, L, h)
                - varphi(-tau_p, L, h)
                - varphi(tau_n, L, h)
        )


def Int5(L_f, L_m, h, tau_n, tau_p):
        return L_m * h / np.sqrt(2*np.pi) * (phi(tau_n, L_f, h) - phi(tau_p, L_f, h)) \
               - 0.5 * L_m * L_f * h**2 * Psi(L_f, h, tau_n, tau_p)


def Int6(L_f, L_m, h, tau_n, tau_p):
        return L_m * h / np.sqrt(2*np.pi) * (2 - phi(tau_p, -L_f, h)
                                             - phi(tau_n, L_f, h)) \
               + 0.5 * L_m * L_f * h**2 * Zeta(L_f, h, tau_n, tau_p)


class BiasEstimator:

        def __init__(self, kernel, design, regression_function, delta_function=None):
                """

                :param kernel:
                :type kernel: Kernel
                :param design: Design
                :type design: DensityFunction
                :param regression_function: lipschits of the regression function
                :type regression_function: Function
                """
                self.k = kernel
                self.h = kernel.get_bandwidth()
                self.f_x = design
                self.m = regression_function
                self.delta_function = delta_function

        def __call__(self, X):
                pass


class BiasTosatto(BiasEstimator):

        def __init__(self, kernel, design, regression_function, delta_function=None):
                """

                :param kernel:
                :type kernel: Kernel
                :param design: Design
                :type design: DensityFunction
                :param regression_function: lipschits of the regression function
                :type regression_function: Function
                """
                super().__init__(kernel, design, regression_function, delta_function=delta_function)

        def __call__(self, X):

                x = X.T

                d, n = x.shape

                local_lipschitz = self.m.local_lipschitz(x)
                log_local_lipschitz = self.f_x.log_lipschitz(x)
                Upsilon = self.f_x.domain()
                L_m = np.repeat([local_lipschitz.L], d, axis=0)
                L_f = np.repeat([log_local_lipschitz.L], d, axis=0)
                delta_p, delta_n = log_local_lipschitz.domain.d_p-x, log_local_lipschitz.domain.d_n-x
                gamma_p, gamma_n = local_lipschitz.domain.d_p-x, local_lipschitz.domain.d_n-x
                # By definition Gamma \subseteq Delta
                gamma_p, gamma_n = np.minimum(delta_p, gamma_p), np.maximum(gamma_n, delta_n)

                u_p, u_n = Upsilon.d_p, Upsilon.d_n

                M = self.m.max_diffference()
                phi_p, phi_n = np.minimum(np.abs(gamma_p), M/L_m), np.minimum(np.abs(gamma_n), M/L_m)

                h = np.repeat([self.h], n, axis=0).T

                den = np.prod(Psi(L_f, h, delta_n, delta_p), axis=0)
                zeta = Zeta(L_f, h, -phi_n, phi_n)
                zeta = np.repeat(np.prod(zeta, axis=0, keepdims=True), d, axis=0)/zeta
                A_k = 2 * Int6(L_f, L_m, h, -phi_n, phi_p)
                assert np.all(A_k >= 0.), (-phi_n, phi_p)
                A = np.sum(A_k * zeta, axis=0)

                B = np.prod(Zeta(L_f, h, gamma_n, gamma_p), axis=0) - np.prod(Zeta(L_f, h, -phi_n, phi_p), axis=0)

                C = 2**d - np.prod(Psi(0., h, gamma_n, gamma_p), axis=0)

                if M == np.infty:
                     ret = A/den
                else:
                     ret = (A + M * (B + C)) / den
                if X.shape[0]==1:
                     ret[np.ravel(x) <= u_n[0]] = np.nan
                     ret[np.ravel(x) >= u_p[0]] = np.nan
                     den[np.ravel(x) <= u_n[0]] = np.nan
                     C[np.ravel(X) <= u_n[0]] = 0.
                     C[np.ravel(X) >= u_p[0]] = 0.
                print("Maximum", np.max(B), np.max(C))
                return ret


class BiasRosenblatt(BiasEstimator):

        def __init__(self, kernel, design, regression_function):
                """

                :param kernel:
                :type kernel: Kernel
                :param design: Design
                :type design: DensityFunction
                :param regression_function: lipschits of the regression function
                :type regression_function: Function
                """
                super().__init__(kernel, design, regression_function)

        def __call__(self, X):
                # in our case  \int_{-\infty}^{+\infty} uK(U)\de u = 1.
                return (0.5 * self.m.second_derivate(X)
                       + self.m.first_derivate(X) * self.f_x.first_derivate(X)/self.f_x.density(X)) \
                       * self.k.get_bandwidth()[0]**2