import numpy as np

from core.kernels import Kernel


class Regression:

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class KernelRegression(Regression):

    def __init__(self, kernel):
        """
        (weighted) Nadaraya-Watson Kernel Regression.
        The weights are needed to represent low-density regions.
        :param kernel:
        :type kernel: Kernel
        """
        self.kernel = kernel
        self._X = None
        self._y = None
        self._w = None

    def fit(self, X, y, w=None):
        k = self.kernel.get_dims()
        assert len(X.shape) == 2, "X must be 2-dimensional."
        assert X.shape[1] == k, "The number of columns of X must match the dimensionality of the kernel."
        assert y.shape[0] == X.shape[0], "The number of samples of the input should match the output."

        self._X = np.expand_dims(X, 1)
        self._y = y
        self._w = w

    def predict(self, X):
        """
        Prediction fron Nadaraya-Watson Kernel Regression.
        :param X: n x d array
        :return:
        """

        X_query = np.expand_dims(X, axis=0)
        if self._w is None:
            k = self.kernel(self._X - X_query)
        else:
            k = self.kernel(self._X - X_query) * self._w.reshape(-1, 1)

        nad_wats = k.T @ self._y / np.sum(k, axis=0, keepdims=True)
        return np.squeeze(nad_wats, axis=0)