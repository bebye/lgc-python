import numpy as np


def create_spirals(n_points=1000, noise=0.5):
    """
    Generate two spirals
    :param n_points
    :param noise
    :return: two spirals dataset
    """

    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    dx = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    dy = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    return (np.vstack((np.hstack((dx, dy)), np.hstack((-dx, -dy)))),
            np.hstack([np.ones(n_points), np.full(n_points, 2)]))
