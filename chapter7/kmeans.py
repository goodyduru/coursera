import numpy as np
import scipy.io as matio
from decimal import *
from matplotlib import pyplot
from skimage import io


def find_closest_centroids(input, centroids):
    K = centroids.shape[0]
    m = input.shape[0]
    idx = np.ones((m, 1), np.int8) * 10000000000000
    i = 0
    while i < m:
        minimum = Decimal("Infinity")
        j = 0
        while j < K:
            dist = np.linalg.norm((input[i, :] - centroids[j, :]))
            if dist < minimum:
                minimum = dist
                idx[i] = j
            j += 1
        i += 1
    return idx


def compute_centroids(input, idx, K):
    m, n = input.shape
    centroids = np.zeros((K, n))
    i = 0
    while i < K:
        Y = np.where( idx == i )
        u = input[Y[0], :]
        centroids[i, :] = np.mean(u, axis=0)
        i += 1
    return centroids


def plot_data_points(input):
    pyplot.scatter(np.ravel(input[:, 0]), np.ravel(input[:, 1]))


def plot_progress_kmeans(input, centroids, previous, idx, K, i):
    plot_data_points(input)
    pyplot.plot(centroids[:, 0], centroids[:, 1], 'kx', ms=10, lw=3)



def run_kmeans(input, centroids, iter):
    (m, n) = input.shape
    K = len(centroids)
    previous_centroids = centroids
    idx = np.zeros((m, 1), dtype=np.int8)
    i = 0
    pyplot.figure(1)
    while i < iter:
        print(i, "iteration")
        idx = find_closest_centroids(input, centroids)
        #plot_progress_kmeans(input, centroids, previous_centroids, idx, K, i)
        #pyplot.show(block=False)
        centroids = compute_centroids(input, idx, K)
        i += 1
    return centroids, idx


def kmeans_init_centroids(input, K):
    randidx = np.random.permutation(input.shape[0])
    centroids = X[randidx[0:K], :]
    return centroids

file = matio.loadmat("ex7data2.mat")
X = np.matrix(file["X"])
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
index = find_closest_centroids(X, initial_centroids)
print(index[0:3])

print("Finding Centroid Means")
Clusters = 3
result_centroids = compute_centroids(X, index, Clusters)
print(result_centroids)

print("\n\nRunning K-Means")
Clusters = 3
max_iter = 10
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
result_centroids, index = run_kmeans(X, initial_centroids, max_iter)
print(result_centroids)

print("\nRunning K-means on image\n")
A = np.array(io.imread("bird_small.png"))
A = A/255
X = np.reshape(A, (A.shape[0] * A.shape[1], 3), order='F')
Clusters = 16
max_iter = 10
initial_centroids = kmeans_init_centroids(X, Clusters)
result_centroids, index = run_kmeans(X, initial_centroids, max_iter)

print("\nApplying K-means to compressing the image\n")
index = find_closest_centroids(X, result_centroids)
print(result_centroids)
X_recovered = result_centroids[index, :]
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], 3), order='F')
pyplot.subplot(121)
pyplot.imshow(A)
pyplot.title("Original")
pyplot.subplot(122)
pyplot.imshow(X_recovered)
pyplot.title("Compressed Image")
pyplot.show()