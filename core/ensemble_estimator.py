import numpy as np

from .kernel_regression import Regression
from .sampler import Sampler


class Ensemble(Regression):

    def __init__(self, regression_constructor, sampler, n_samples=10000, n_models=10):
        """
        This class implements the "Ensamble Regression" and it is needed to estimate the average of the estimator,
        which is needed in the computation of the bias.
        :param regression:
        :param sampler:
        :type sampler: Sampler
        """
        self.n_samples = n_samples
        self.regressions = [regression_constructor() for _ in range(n_models)]
        self.max_min = -np.inf
        self.min_max = np.inf
        for reg in self.regressions:
            X, y, w = sampler(n_samples)

            min_ = np.min(X)
            max_ = np.max(X)
            if min_ > self.max_min:
                self.max_min = min_
            if max_ < self.min_max:
                self.min_max = max_
            reg.fit(X, y, w)

    def predict(self, X):
        return np.mean([reg.predict(X) for reg in self.regressions], axis=0)

