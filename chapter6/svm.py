import numpy as np
import scipy.io as matio
import matplotlib.pyplot as pyplot
from sklearn import svm


def plot_data(x, y):
    pos = np.where([y == 1])
    neg = np.where([y == 0])
    pyplot.plot(x[pos[1], 0], x[pos[1], 1], "k+", x[neg[1], 0], x[neg[1], 1], "yo", ms=7, lw=1)
    pyplot.show()


def linear_kernel(x, y):
    x = np.reshape(x, (np.size(x), 1))
    y = np.reshape(y, (np.size(y), 1))
    matrix = x.T.dot(y)
    return matrix


def gaussian_kernel(x1, x2, sigma):
    x1 = np.reshape(x1, (np.size(x1), 1))
    x2 = np.reshape(x2, (np.size(x2), 1))
    sim = np.exp(-np.sum(np.power((x1 - x2), 2)) / (2 * (sigma ** 2)))
    return sim


def visualise_boundary_model(x, y, lin, title):
    color_map = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)}
    x_min, x_max = x[:, 0].min(), x[:, 0].max()
    y_min, y_max = x[:, 1].min(), x[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = lin.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    pyplot.contourf(xx, yy, Z, cmp=pyplot.cm.Paired)
    pyplot.axis("off")
    colors = [color_map[i] for i in y]
    pyplot.scatter(x[:, 0], x[:, 1], c=colors, edgecolors='black')
    pyplot.title(title)
    pyplot.show()

file = matio.loadmat("ex6data1.mat")
input, output = file["X"], np.ravel(file["y"])

plot_data(input, output)

C = 1;
lin = svm.SVC(kernel="linear", C=C, tol=1E-3, max_iter=20)
clf = lin.fit(input, output)
visualise_boundary_model(input, output, clf, "Decision Boundary For Linear")
t1, t2, sig = np.array([1, 2, 1]), np.array([0, 4, -1]), 2
sim = gaussian_kernel(t1, t2, sig)
print(sim)

file = matio.loadmat("ex6data2.mat")
input, output = file["X"], np.ravel(file["y"])
plot_data(input, output)

C, gamma = 1, 0.1
lin = svm.SVC(kernel="rbf", C=C, tol=1E-3, max_iter=5)
clf = lin.fit(input, output)
visualise_boundary_model(input, output, clf, "Decision Boundary For Gaussian")