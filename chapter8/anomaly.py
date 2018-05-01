import numpy as np
import scipy.io as matio
import matplotlib.pyplot as pyplot
import math


def estimate_gaussian(input_matrix):
    mean = np.mean(input_matrix, axis=0)
    variance = np.var(input_matrix, axis=0)
    return np.matrix(mean), np.matrix(variance)


def multivariate_gaussian(input_matrix, mean, variance):
    input_matrix = input_matrix - mean
    k = np.size(mean)
    if variance.shape[1] == 1 or variance.shape[0] == 1:
        variance = np.diag(np.array(variance).flatten())
    gaussian = ((2 * math.pi)**(-k / 2)) * (np.linalg.det(variance) ** (-0.5)) * \
        np.exp(-0.5 * np.sum(np.multiply(input_matrix * np.linalg.pinv(variance), input_matrix), axis=1))
    return gaussian


def visualise_fit(input_matrix, mean, variance):
    x1, x2 = np.meshgrid(np.arange(0, 35, .5), np.arange(0, 35, .5))
    x1, x2 = np.matrix(x1), np.matrix(x2)
    x_result = np.concatenate((x1.flatten().T, x2.flatten().T), axis=1)
    z = multivariate_gaussian(x_result, mean, variance)
    z = np.reshape(z, x1.shape)
    pyplot.plot(input_matrix[:, 0], input_matrix[:, 1], 'bx')
    exp = np.matrix(np.power(10.0, np.arange(-20, 0, 3))).T
    pyplot.contour(x1, x2, z, exp)


def select_threshold(output_cv, probability):
    best_f1_score = 0
    best_epsilon = 0
    minimum = np.min(probability)
    maximum = np.max(probability)
    step_size = (maximum - minimum) / 1000

    for eps in np.arange(minimum, maximum, step_size):
        cv_predictions = np.where((probability < eps), 1, 0)
        true_positive = np.sum(np.where(np.logical_and((cv_predictions == 1), (output_cv == 1)), 1, 0))
        false_positive = np.sum(np.where(np.logical_and((cv_predictions == 1), (output_cv == 0)), 1, 0))
        true_negative = np.sum(np.where(np.logical_and((cv_predictions == 0), (output_cv == 1)), 1, 0))
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + true_negative)
        f1_score = (2 * precision * recall) / (precision + recall)

        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_epsilon = eps

    return best_epsilon, best_f1_score


file = matio.loadmat("ex8data1.mat")
X, X_val, y_val = np.matrix(file["X"]), np.matrix(file["Xval"]), np.matrix(file["yval"])

print("Visualizing dataset for outlier detection", end="\n")
pyplot.plot(X[:, 0], X[:, 1], 'bx')
pyplot.axis([0, 30, 0, 30])
pyplot.xlabel("Latency (ms)")
pyplot.ylabel("Throughput (mb/s)")
pyplot.show()

print("Visualizing Gaussian fit", end="\n")
mu, sigma2 = estimate_gaussian(X)
p = multivariate_gaussian(X, mu, sigma2)
visualise_fit(X, mu, sigma2)
pyplot.show()

print("Selecting threshold", end="\n")
p_val = multivariate_gaussian(X_val, mu, sigma2)
epsilon, f_1 = select_threshold(y_val, p_val)
print(epsilon, f_1, end="\n")
outliers = np.where((p < epsilon))
visualise_fit(X, mu, sigma2)
pyplot.plot(X[outliers[0], 0], X[outliers[0], 1], 'ro', lw=2, ms=10)

print("Multidimensional outlier", end="\n")
file = matio.loadmat("ex8data2.mat")
X, X_val, y_val = file["X"], file["Xval"], file["yval"]
mu, sigma2 = estimate_gaussian(X)
p = multivariate_gaussian(X, mu, sigma2)
p_val = multivariate_gaussian(X_val, mu, sigma2)
epsilon, f_1 = select_threshold(y_val, p_val)
print(epsilon)
print(f_1)
print(np.sum(np.where((p < epsilon), 1, 0)))