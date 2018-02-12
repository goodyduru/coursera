import numpy as np
import scipy.io as matio
import matplotlib.pyplot as pyplot
import scipy.optimize as optimize


def linear_reg_cost_function(weight, x, y, reg):
    (m, n) = np.shape(x)
    weight = weight.reshape((n, 1))
    predictions = np.dot(x, weight)
    diff = predictions - y
    sqr_errors = np.power(diff, 2)
    cost = np.sum(sqr_errors) / (2 * m) + (reg * np.sum(np.power(weight[1:], 2))) / (2 * m)
    return cost


def linear_reg_gradient(weight, x, y, reg):
    (m, n) = np.shape(x)
    weight = weight.reshape((n, 1))
    predictions = np.dot(x, weight)
    diff = predictions - y
    error = np.multiply(diff, x)
    s = np.sum(error, axis=0) / m
    grad = s.T + (reg/m * weight)
    grad[0] = grad[0] - (reg/m * weight[0])
    return np.squeeze(np.asarray(grad))


def train_lin_reg(x, y, reg):
    shape = np.shape(x)
    initial_theta = np.zeros([shape[1], 1])
    result = optimize.minimize(linear_reg_cost_function, initial_theta, args=(x, y, reg),
                               method='CG', jac=linear_reg_gradient, options={'disp': True, 'maxiter': 200})
    return result.x


def learning_curve(x, y, xval, yval, reg):
    m = len(x)
    error_train = np.zeros([m, 1])
    error_val = np.zeros([m, 1])
    i = 1
    while i <= m:
        weight = train_lin_reg(x[0:i, :], y[0:i], reg)
        error_train[i - 1] = linear_reg_cost_function(weight, x[0:i, :], y[0:i], reg)
        error_val[i - 1] = linear_reg_cost_function(weight, xval, yval, reg)
        i += 1
    return error_train, error_val


def poly_feature(x, p):
    out = np.zeros([len(x), p])
    i = 0
    while i < p:
        out[:, i] = np.power(x, i + 1).flatten()
        i += 1
    return out


def feature_normalize(x):
    mu = np.mean(x, 0)
    sigma = np.std(x, 0)
    x_norm =  np.divide((x - mu), sigma)
    return (x_norm, mu, sigma)


def plot_fit(min_x, max_x, mu, sigma, weight, p):
    x = np.arange(min_x - 15, max_x + 25, 0.05).transpose()
    x_poly = poly_feature(x, p)
    x_poly = x_poly - mu
    x_poly = x_poly / sigma
    x_poly = np.concatenate((np.ones([len(x), 1]), x_poly), axis=1)
    return x, np.dot(x_poly, weight)


def validation_curve(x, y, xval, yval):
    lbd_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.01, 0.1, 0.3, 1, 3, 10]
    shape = np.shape(x)
    error_train = np.zeros([len(lbd_vec), 1])
    error_val = np.zeros([len(lbd_vec), 1])
    for k, v in enumerate(lbd_vec):
        weight = train_lin_reg(x, y, v).reshape([shape[1], 1])
        error_train[k] = linear_reg_cost_function(weight, x, y, 0)
        error_val[k] = linear_reg_cost_function(weight, xval, yval, 0)
    return lbd_vec, error_train, error_val


file = matio.loadmat("ex5data1.mat")
X, y, Xval = np.matrix(file["X"]), np.matrix(file["y"]), np.matrix(file["Xval"])
yval, Xtest, ytest = np.matrix(file["yval"]), np.matrix(file["Xtest"]), np.matrix(file["ytest"])
m = len(X)

# First Plot
pyplot.plot(X, y, 'rx', ms=10, lw=1.5)
pyplot.xlabel("Change in water level(x)")
pyplot.ylabel("Water flowing out of the dam(y)")
pyplot.show()

theta = np.matrix([[1], [1]])
full_x = np.concatenate((np.ones([m, 1]), X), axis=1)
J = linear_reg_cost_function(theta, full_x, y, 1)
print("Cost at theta = [1 ; 1]", J)
grad = linear_reg_gradient(theta, full_x, y, 1)
print("Gradient at theta=[1 ; 1]", grad[0], grad[1])
shape = np.shape(full_x)
result = train_lin_reg(full_x, y, 0).reshape([shape[1], 1])

#Second Plot
pyplot.plot(X, y, 'rx', X, full_x * result, '--', ms=10, lw=1.5 )
pyplot.xlabel("Change in water level(x)")
pyplot.ylabel("Water flowing out of the dam(y)")
pyplot.show()

full_xval = np.concatenate((np.ones([len(Xval), 1]), Xval), axis=1)
err_train, err_val = learning_curve(full_x, y, full_xval, yval, 0)
pyplot.plot(range(0, m), err_train, range(0, m), err_val)
pyplot.title("Learning curve for linear regression")
pyplot.legend(["Train", "Cross Validation"])
pyplot.xlabel("Number of training examples")
pyplot.ylabel("Error")
pyplot.axis([0, 13, 0, 150])
pyplot.show()

"""
Polynomial Regression
"""

X_poly = poly_feature(X, 8)
X_poly, mu, sigma = feature_normalize(X_poly)
X_poly = np.concatenate((np.ones([len(X_poly), 1]), X_poly), axis=1)

X_poly_test = poly_feature(Xtest, 8)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.concatenate((np.ones([len(X_poly_test), 1]), X_poly_test), axis=1)

X_poly_val = poly_feature(Xval, 8)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.concatenate((np.ones([len(X_poly_val), 1]), X_poly_val), axis=1)

lbd = 3
shape = np.shape(X_poly)
theta = train_lin_reg(X_poly, y, lbd).reshape([shape[1], 1])
pyplot.figure(1)
new_x, new_x_poly = plot_fit(np.min(X), np.max(y), mu, sigma, theta, 8)
pyplot.plot(X, y, 'rx', new_x, new_x_poly, '--', lw=1.5, ms=10)
pyplot.xlabel("Change In Water Level (x)")
pyplot.ylabel("Water flowing out of the dam")
pyplot.title("Polynomial Regression Fit (lambda = {0})".format(lbd))

pyplot.figure(2)
err_train, err_val = learning_curve(X_poly, y, X_poly_val, yval, lbd)
pyplot.plot(range(1, m + 1), err_train, range(1, m + 1), err_val)
pyplot.title("Polynomial Regressional Curve")
pyplot.legend(["Train", "Cross Validation"])
pyplot.xlabel("Number of training examples")
pyplot.ylabel("Error")
pyplot.axis([0, 13, 0, 150])
pyplot.show()

lbd_v, err_train, err_val = validation_curve(X_poly, y, X_poly_val, yval)
pyplot.plot(lbd_v, err_train, lbd_v, err_val)
pyplot.legend(["Train", "Cross Validation"])
pyplot.xlabel("Lambda")
pyplot.ylabel("Error")
pyplot.show()