from .kernels import Kernel
from .ensemble_estimator import Ensemble
from .functions import DensityFunction, Function
from .kernel_regression import KernelRegression
from .sampler import Sampler
from .bias_estimation import BiasTosatto, BiasRosenblatt

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rc('hatch', color='k', linewidth=0.3)


class ExperimentResult:

    def __init__(self, x_grid, y_design, y_true, y_pred, b_tosatto, b_rosenblatt, x_min, x_max):
        self.x_grid = x_grid
        self.y_design = y_design
        self.y_true = y_true
        self.y_pred = y_pred
        self.b_tosatto = b_tosatto
        self.b_rosenblatt = b_rosenblatt
        self.x_min = x_min
        self.x_max = x_max


class Experiment:

    def __init__(self, regression_function, design, kernel, n_samples=10000, n_models=100, interval=None,
                 delta_function=None):
        """
        This experiments numerically estimates the true bias, and compares with the bounds given by Tosatto et. al
        and Rosenblatt.
        :param regression_function:
        :type regression_function: Function
        :param design:
        :type design: DensityFunction
        :param kernel:
        :type kernel: Kernel
        :param n_samples:
        :type n_samples: int
        :param n_models:
        :type n_models: int
        """
        self.m = regression_function
        self.f_x = design
        self.k = kernel
        self.regr = Ensemble(lambda: KernelRegression(kernel), Sampler(self.f_x, self.m, fixed_interval=interval),
                             n_samples=n_samples, n_models=n_models)
        self.bias_tosatto = BiasTosatto(kernel, design, regression_function, delta_function=delta_function)
        self.bias_rosenblatt = BiasRosenblatt(kernel, design, regression_function)

    def __call__(self, x_grid):
        y_true = self.m(x_grid, noise=False)
        y_pred = self.regr.predict(x_grid.reshape(-1, 1))
        b_tosatto = self.bias_tosatto(x_grid.reshape(-1, 1))
        b_rosenblatt = np.ravel(self.bias_rosenblatt(x_grid.reshape(-1, 1)))
        x_min = self.regr.max_min
        x_max = self.regr.min_max
        y_design = self.f_x.density(x_grid)
        return ExperimentResult(x_grid, y_design, y_true, y_pred, b_tosatto, b_rosenblatt, x_min, x_max)

