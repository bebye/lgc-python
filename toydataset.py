import numpy as np
import numpy.random as random
from sklearn.utils import shuffle as util_shuffle


def create_spirals(n_points=1000, n_rotation=2):
    """
    Generate two spirals
    :param n_points: the number of points per class
    :param n_rotation: the number of rotation
    :return: two spirals dataset
    """

    n_class = 2
    theta = 2 * np.pi * np.linspace(0, 1, n_class + 1)[:n_class]
    X = []

    for c in range(n_class):
        t_shift = theta[c]

        t = n_rotation * np.pi * 2 * np.sqrt(random.rand(1, n_points))
        x = t * np.cos(t + t_shift)
        y = t * np.sin(t + t_shift)
        Xc = np.concatenate((x, y))
        X.append(Xc.T)

    X = np.concatenate(X)
    Y = np.hstack([np.zeros(n_points), np.ones(n_points)])

    X, Y = util_shuffle(X, Y)

    return X, Y
