import numpy as np
import scipy.io as matio
import matplotlib.pyplot as pyplot
from sklearn import svm
from decimal import *


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


def gaussian_kernel(x1, x2, sigma=0.1):
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


def data_set_parameters(x, y, x_val, y_val):
    c_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    gamma_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    i, c, g = 0, 0, 0
    error = Decimal('+Infinity')
    while i < len(c_list):
        j = 0
        while j < len(gamma_list):
            svc = svm.SVC(kernel="rbf", C=c_list[i], tol=1E-3, gamma=gamma_list[j])
            svc.fit(x, y)
            predictions = svc.predict(x_val)
            mean_error = np.mean(np.where((y_val != predictions), 1, 0))
            if mean_error < error:
                c = c_list[i]
                g = gamma_list[j]
            j += 1
        i += 1
    return c, g


file = matio.loadmat("ex6data1.mat")
input, output = file["X"], np.ravel(file["y"])

plot_data(input, output)

C = 1;
lin = svm.SVC(kernel="linear", C=C, tol=1E-3, max_iter=20)
clf = lin.fit(input, output)
visualise_boundary_model(input, output, clf, "Decision Boundary For Linear")
t1, t2 = np.array([1, 2, 1]), np.array([0, 4, -1])
sim = gaussian_kernel(t1, t2, sigma=2)
print(sim)

file = matio.loadmat("ex6data2.mat")
input, output = file["X"], np.ravel(file["y"])
plot_data(input, output)

C, gamma = 13, 1
lin = svm.SVC(kernel="rbf", C=C, tol=1E-3, gamma=gamma)
clf = lin.fit(input, output)
visualise_boundary_model(input, output, clf, "Decision Boundary For Gaussian")

file = matio.loadmat("ex6data3.mat")
input, output, xval, yval = file["X"], np.ravel(file["y"]), file["Xval"], np.ravel(file["yval"])
plot_data(input, output)

C, gamma = data_set_parameters(input, output, xval, yval)
lin = svm.SVC(kernel="rbf", C=C, tol=1E-3, gamma=gamma)
clf = lin.fit(input, output)
visualise_boundary_model(input, output, clf, "Decision Boundary For RBF")