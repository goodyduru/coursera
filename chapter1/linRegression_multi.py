import numpy as np
import pandas
import matplotlib.pyplot as pyplot

data = pandas.read_csv("ex1data2.txt", header=None)
matrix = np.matrix(data)
x_val = matrix[:, 0:2]
y_val = matrix[:, 2]


def feature_normalize(input_matrix):
    mean = np.mean(input_matrix, 0)
    std = np.std(input_matrix, 0)
    input_norm = np.divide((input_matrix - mean), std)
    return mean, std, input_norm

(mu, sigma, X) = feature_normalize(x_val)
m = len(y_val)
ones = np.ones([m, 1])
X = np.concatenate((ones, X), axis=1)
alpha = 1
num_iter = 400
theta = np.zeros([3, 1])


def compute_cost_multi(input_matrix, output_matrix, weight):
    num_rows = len(output_matrix)
    predictions = input_matrix.dot(weight)
    sqr_errors = np.power((predictions - output_matrix), 2)
    cost = sum(sqr_errors) / (2 * num_rows)
    return cost


def gradient_descent_multi(input_matrix, output_matrix, weight, step, iteration):
    num_rows, num_cols = input_matrix.shape
    i = 0
    cost_history = np.zeros((iteration, 1))
    while i < iteration:
        new_weight = weight
        j = 0
        while j < num_cols:
            predictions = input_matrix * weight
            errors = predictions - output_matrix
            new_weight[j] = weight[j] - ((step / num_rows) * np.sum(np.multiply(errors, input_matrix[:, j])))
            j += 1
        weight = new_weight
        cost_history[i] = compute_cost_multi(input_matrix, output_matrix, weight)
        i += 1
    return weight, cost_history

theta, J_history = gradient_descent_multi(X, y_val, theta, alpha, num_iter)
pyplot.plot(np.arange(J_history.shape[0]), J_history, '-b', lw=2)
pyplot.show()
size = (1650 - mu[0, 0]) / sigma[0, 0]
room = (3 - mu[0, 1]) / sigma[0, 1]
price = np.dot([1, size, room],  theta)
print("Predicted price of a 1650 sq ft, 3 br house with gradient Descent", price)

X = np.concatenate((ones, x_val), axis=1)


def normal_eqn(input_matrix, output_matrix):
    weight = np.dot(np.dot(np.linalg.pinv(input_matrix.T.dot(input_matrix)), input_matrix.T), output_matrix)
    return weight

theta = normal_eqn(X, y_val)
print("Theta computed from normal Equation", theta)
price = np.dot([1, 1650, 3], theta)
print("Price computed from normal Equation", price)
