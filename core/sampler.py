from .functions import Function, DensityFunction

import numpy as np


class Sampler:

    def __init__(self, design=None, phenomenon=None, fixed_interval=None):
        """

        :param design:
        :type design: DensityFunction
        :param phenomenon:
        :type phenomenon: Function
        :param fixed_interval:
        """

        self.design = design
        self.phenomenon = phenomenon
        self.fixed_interval = fixed_interval

    def __call__(self, n):
        if self.fixed_interval is None:
            X = self.design(n)
            return X, self.phenomenon(X), None
        else:
            X = np.linspace(*self.fixed_interval, n).reshape(-1, 1)
            return X, self.phenomenon(X), self.design.density(X)
