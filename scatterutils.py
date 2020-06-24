import matplotlib.pyplot as plt


def show_scatter_spirals(X, Y):
    color = ['r' if y == 0 else 'b' for y in Y]
    plt.scatter(X[0:, 0], X[0:, 1], c=color)
    plt.show()
