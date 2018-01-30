import numpy as np
import pandas
import matplotlib.pyplot as pyplot

def warmUp():
    a = np.identity(5)
    print(a)

warmUp()
file = pandas.read_csv("ex1data1.csv", header=None)
arr = np.matrix(file)
xval = arr[:, 0]
yval = arr[:, 1]
pyplot.plot(xval, yval, "r+", markersize=10)
pyplot.xlabel("Profit in 10000s")
pyplot.ylabel("Population in 10000s")
pyplot.show()
m = len(yval)
ones = np.ones([m, 1])
X = np.concatenate((ones, xval), axis=1)
theta = np.zeros([2, 1])

iterations = 1500
alpha = 0.01

def computeCost(X, y, theta):
    m = len(y)
    J = 0
    predictions = X * theta
    sqrErrors = np.power((predictions - y), 2)
    J = 1/(2 * m) * np.sum(sqrErrors)
    return J

J = computeCost(X, yval, theta)
print(J)

J = computeCost(X, yval, np.matrix("-1; 2"))
print(J)


def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    i = 0
    while i < iterations:
        predictions = X * theta
        errors = predictions - y
        temp0 = theta[0] - ((alpha/m) * np.sum(errors))
        temp1 = theta[1] - ((alpha/m) * np.sum(np.multiply(errors, X[:, 1])))
        theta[0] = temp0
        theta[1] = temp1
        i += 1
    return theta


theta = gradientDescent(X, yval, theta, alpha, iterations)
print(theta)
predict = np.matrix(np.array([1, 3.5])) * theta
print(predict*10000)
predict = np.matrix("1, 7") * theta
print(predict * 10000)