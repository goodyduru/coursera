import numpy as np
import pandas
import matplotlib.pyplot as pyplot


def warm_up():
    a = np.identity(5)
    print(a)

warm_up()
file = pandas.read_csv("ex1data1.csv", header=None)
arr = np.matrix(file)
x_val = arr[:, 0]
y_val = arr[:, 1]
pyplot.plot(x_val, y_val, "r+", markersize=10)
pyplot.xlabel("Profit in 10000s")
pyplot.ylabel("Population in 10000s")
pyplot.show()
m = len(y_val)
ones = np.ones([m, 1])
X = np.concatenate((ones, x_val), axis=1)
theta = np.zeros([2, 1])

iterations = 1500
alpha = 0.01


def compute_cost(input_matrix, output_matrix, weight):
    num_rows = len(output_matrix)
    predictions = input_matrix * weight
    sqr_errors = np.power((predictions - output_matrix), 2)
    cost = 1/(2 * num_rows) * np.sum(sqr_errors)
    return cost

J = compute_cost(X, y_val, theta)
print(J)

J = compute_cost(X, y_val, np.matrix("-1; 2"))
print(J)


def gradient_descent(input_matrix, output_matrix, weight, step, max_iter):
    num_rows = len(output_matrix)
    i = 0
    while i < max_iter:
        predictions = input_matrix * weight
        errors = predictions - output_matrix
        temp0 = weight[0] - ((step/num_rows) * np.sum(errors))
        temp1 = weight[1] - ((step/num_rows) * np.sum(np.multiply(errors, input_matrix[:, 1])))
        weight[0] = temp0
        weight[1] = temp1
        i += 1
    return weight


theta = gradient_descent(X, y_val, theta, alpha, iterations)
print(theta)
predict = np.matrix(np.array([1, 3.5])) * theta
print(predict*10000)
predict = np.matrix("1, 7") * theta
print(predict * 10000)
