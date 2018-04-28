import scipy.io as matio
import numpy as np
from matplotlib import pyplot
import math


def feature_normalize(input_matrix):
    mean = np.mean(input_matrix, axis=0)
    std_dev = np.std(input_matrix, axis=0)
    input_norm = np.divide((input_matrix - mean), std_dev)
    return mean, std_dev, input_norm


def pca(input_matrix):
    m = len(input_matrix)
    covariance_matrix = (input_matrix.T * input_matrix)/m
    unitary_matrix, singular_value, v = np.linalg.svd(covariance_matrix)
    return unitary_matrix, singular_value


def draw_line(p1, p2, *args, **keywordargs):
    pyplot.plot([p1[0, 0], p2[0, 0]], [p1[0, 1], p2[0, 1]], args, keywordargs)


def project_data(input_matrix, unitary_matrix, dimension):
    reduced_matrix = input_matrix * unitary_matrix[:, 0:dimension]
    return reduced_matrix


def recover_data(reduced_matrix, unitary_matrix, dimension):
    recovered_matrix = reduced_matrix * unitary_matrix[:, 0:dimension].T
    return recovered_matrix


def display_data(input_matrix):
    (m, n) = input_matrix.shape
    width = math.floor(math.sqrt(n))
    height = math.floor(n / width)

    rows = math.floor(math.sqrt(m))
    cols = math.ceil(m / rows)

    """
        width 32
        height 32
        rows 10
        cols 10
    """

    pad = 1
    display_array = -np.ones([pad + rows * (height + pad), pad + cols * (width + pad)])
    current_row, current_example = 0, 0
    while current_row < rows:
        current_col = 0
        while current_col < cols:
            max_value_in_col = np.max(np.abs(input_matrix[current_example, :]))
            square_row = pad + current_row * (height + pad) + np.arange(height)
            square_col = pad + current_col * (width + pad) + np.arange(width)
            image_patch = np.reshape(input_matrix[current_example, :], (height, width)) / max_value_in_col
            x, y = np.meshgrid(square_row, square_col)
            display_array[x, y] = image_patch
            current_example += 1
            current_col += 1
        current_row += 1
    pyplot.imshow(display_array)

print("Visualizing data set for PCA\n")
file = matio.loadmat('ex7data1.mat')
X = np.matrix(file["X"])
pyplot.plot(X[:, 0], X[:, 1], 'bo')
pyplot.axis([0.5, 6.5, 2, 8])
print("Running PCA on example dataset. \n")
mu, sigma, X_norm = feature_normalize(X)
U, S = pca(X_norm)
draw_line(mu, mu + 1.5 * S[0] * U[:, 0].T, '-k', lw=2)
draw_line(mu, mu + 1.5 * S[1] * U[:, 1].T, '-k', lw=2)
pyplot.show()
print("Top eigenvector:", end="\n")
print(U[0, 0], U[1, 0], end="\n")

print("Dimension Reduction on Example Dataset \n")
pyplot.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
pyplot.axis([-4, 3, -4, 3])
pyplot.show()

K = 1
Z = project_data(X_norm, U, K)
print("Shape Of Z", np.shape(Z), end="\n")
print("Projection Of The First Example", Z[0], end="\n")

X_rec = recover_data(Z, U, K)
print("Approximation of the First Example", X_rec[0, 0], X_rec[0, 1], end="\n")

pyplot.plot(X_rec[:, 0], X_rec[:, 1], 'ro')
index = 0
while index < len(X_rec):
    draw_line(X_norm[index, :], X_rec[index, :], '--k', lw=1)
    index += 1
pyplot.show()

print("Loading Face Dataset")
file = matio.loadmat("ex7faces.mat")
X = np.matrix(file["X"])
display_data(X[0:100, :])
pyplot.show()

mu, sigma, X_norm = feature_normalize(X)
U, S = pca(X_norm);
display_data(U[:, 0:36].T)
pyplot.show()
K = 100
Z = project_data(X_norm, U, K)
print("Shape of Z", Z.shape, end="\n")
X_rec = recover_data(Z, U, K)

pyplot.subplot(121)
display_data(X_norm[0:100, :])
pyplot.title("Original Faces")

pyplot.subplot(122)
display_data(X_rec[0:100, :])
pyplot.title("Recovered Faces")
pyplot.show()
