import numpy as np
import pandas
import matplotlib

def warmUp():
    a = np.identity(5)
    print(a)

warmUp()
file = pandas.read_csv("ex1data1.csv", header=None)
arr = np.matrix(file)
xval = arr[:, 0]
yval = arr[:, 1]
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

data = pandas.read_csv("ex1data2.txt", header=None)
matrix = np.matrix(data)
xval = matrix[:, 0:2]
yval = matrix[:, 2]


def featureNormalize(X):
    X_norm = X
    mu = np.zeros([1, np.shape(X)[1]])
    sigma = np.zeros([1, np.shape(X)[1]])

    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_norm =  np.divide((X - mu), sigma)
    return (mu, sigma, X_norm)

(mu, sigma, X) = featureNormalize(xval)
#print(mu, sigma)
#print(X[0:10])
m = len(yval)
ones = np.ones([m, 1])
X = np.concatenate((ones, X), axis=1)
alpha = 1
num_iters = 400
theta = np.zeros([3, 1])

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(X)
    n = np.shape(X)[1]
    i = 0
    while i < num_iters:
        thetaNew = theta
        j = 0
        while j < n:
            predictions = X * theta
            errors = predictions - y
            thetaNew[j] = theta[j] - ((alpha / m) * np.sum(np.multiply(errors, X[:, j])))
            j += 1
        theta = thetaNew
        i += 1
    return theta

theta = gradientDescentMulti(X, yval, theta, alpha, num_iters)
print(theta)
size = (1650 - mu[0, 0]) / sigma[0, 0]
room = (3 - mu[0, 1]) / sigma[0,1]
price = np.dot([1, size, room],  theta)
print("Predicted price of a 1650 sq ft, 3 br house with gradient Descent", price)

X = np.concatenate((ones, xval), axis=1)

def normalEqn(X, y):
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
    return theta

theta = normalEqn(X, yval)
print("Theta computed from normal Equation", theta)
price = np.dot([1, 1650, 3], theta)
print("Price computed from normal Equation", price)