import numpy as np


def distmat(x, y):
    return np.sum(x**2, 0)[:, None] + np.sum(y**2, 0)[None, :] - 2*x.transpose().dot(y)

