import numpy as np
import scipy.io as matio
import matplotlib.pyplot as pyplot
import scipy.optimize as optimize


def linear_reg_cost_function(weight, x, y, reg):
    (m, n) = np.shape(x)
    weight = weight.reshape((n, 1))
    predictions = x * weight
    diff = predictions - y
    sqr_errors = np.power(diff, 2)
    cost = np.sum(sqr_errors) / (2 * m) + (reg * np.sum(np.power(weight[1:], 2))) / (2 * m)
    return cost


def linear_reg_gradient(weight, x, y, reg):
    (m, n) = np.shape(x)
    weight = weight.reshape((n, 1))
    predictions = x * weight
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

