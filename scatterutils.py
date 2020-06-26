import matplotlib.pyplot as plt


def show_scatter_spirals(x, y):
    plt.scatter(x[y == 0, 0], x[y == 0, 1], c='gray', marker='x', label='unlabeled')
    plt.scatter(x[y == 1, 0], x[y == 1, 1], c='red', marker='o', label='class1')
    plt.scatter(x[y == 2, 0], x[y == 2, 1], c='blue', marker='o', label='class2')
    plt.show()
