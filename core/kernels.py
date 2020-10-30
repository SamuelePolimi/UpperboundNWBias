import numpy as np


class Kernel:

    def get_dims(self):
        pass

    def get_bandwidth(self):
        pass

    def __call__(self, *args, **kwargs):
        pass


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
        mahalanobis = np.swapaxes(_delta, n_dim-1, n_dim) @ self.precision @ _delta/2.

        return np.squeeze(np.exp(-mahalanobis)/np.sqrt((2. * np.pi)**self.k / np.prod(self.h**2)), axis=(n_dim-1, n_dim))

    def get_dims(self):
        return self.k

    def get_bandwidth(self):
        return self.h
