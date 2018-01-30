import numpy as np
import pandas
import matplotlib

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