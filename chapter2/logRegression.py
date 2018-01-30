import numpy as np
import pandas
import scipy.optimize as optimize
import matplotlib.pyplot as pyplot
import matplotlib.lines as lines
data = pandas.read_csv("ex2data1.txt", header=None)
matrix = np.matrix(data)

xval = matrix[:, 0:2]
yval = matrix[:, 2]

(m, n) = np.shape(xval)
ones = np.ones([m, 1])
X = np.concatenate((ones, xval), axis=1)
initial_theta = np.zeros([n + 1, 1])

def sigmoid(z):
    g = np.divide(1, (1 + np.exp(-z)))
    return g


def costFunction(theta, X, y):
    (m, n) = np.shape(X)
    theta = theta.reshape((n, 1))
    grad = np.zeros(np.shape(theta))
    my_x = sigmoid(np.transpose(X.dot(theta)))
    term1 = np.log(my_x)
    term2 = np.log(1 - my_x)
    J = ( np.dot(term1, -y) - np.dot(term2, (1 - y))) / m
    grad = np.transpose(np.dot((my_x - np.transpose(y)), X)) / m
    return J, grad

def Gradient(theta, X, y):
    (m, n) = np.shape(X)
    theta = theta.reshape((n, 1))
    my_x = sigmoid(np.transpose(X.dot(theta)))
    return grad

cost, grad = costFunction(initial_theta, X, yval)
print("Cost at initial theta(zeros)", cost)
print("Gradient at initial theta", grad)

test_theta = np.matrix("-24; 0.2; 0.2")
cost, grad = costFunction(test_theta, X, yval)
print("Cost at test theta", cost)
print("Gradient at test theta", grad)

result = optimize.fmin_tnc(costFunction, x0=initial_theta, args=(X, yval) )
theta = np.matrix(result[0])
theta = theta.reshape((n + 1, 1))
task = np.matrix("1; 45; 85")
prob = sigmoid(np.matrix("1 45 85") * theta)
print("Probability of admission with scores between 45 and 85", prob)

def plotData(theta, X, y):
    pos = np.where((y == 1))[0]
    neg = np.where((y == 0))[0]
    pyplot.figure(1)
    pyplot.subplot(211)
    pyplot.plot(X[pos, 1], X[pos, 2], "k+", markersize=10)
    pyplot.plot(X[neg, 1], X[neg, 2], "ko", markersize=7)

def mapFeature(X1, X2):
    degree = 6
    out = np.ones([len(X1[:, 0]), 1])
    i = 1
    while i <= degree:
        j = 0
        while j <= i:
            out = np.concatenate((out, np.multiply(np.power(X1, (i - j )), np.power(X2, j))), axis=1)
            j += 1
        i += 1
    return out

def plotDescisionBoundary(theta, X, y):
    plotData(theta, X, y)
    if ( np.shape(X)[1] > 3 ):
        u = np.transpose(np.matrix(np.arange(-1, step = 1.5, stop = 50)))
        v = np.transpose(np.matrix(np.arange(-1, step = 1.5, stop=50)))

        z = np.zeros([len(u), len(v)])

        i = 0
        while i < len(u):
            j = 0
            while j < len(v):
                z[i, j] = np.dot(mapFeature(u[i], v[j]), theta )
                j += 1
            i += 1
        z = z.transpose()
        pyplot.contour(u, v, z, [0, 0])
        pyplot.show()
    else:
        plot_x = np.matrix([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])
        plot_y = (-1/theta[2]) * (theta[1] * plot_x + theta[0])
        pyplot.subplot(212)
        pyplot.plot(plot_x[0, 0], plot_y[0, 0],  lw=2, ls='--')
        pyplot.axes([30, 100, 30, 100])
        pyplot.ylabel('Decision Boundary')
        pyplot.show()

plotDescisionBoundary(theta, X, yval)