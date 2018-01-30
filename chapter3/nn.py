import scipy.io as matio
import numpy as np


def sigmoid(z):
    g = np.divide(1, (1 + np.exp(-z)))
    return g


def predict(Theta1, Theta2, X, y):
    X = np.concatenate((np.ones([len(X), 1]), X), axis=1)
    A_1 = np.zeros([len(Theta1), 1])
    A_2 = np.zeros([len(Theta2), 1])
    print("Size of Theta1", np.shape(Theta1), "Size of X", np.shape(X))
    A_1 = sigmoid(Theta1@X.T)
    (f, g) = np.shape(A_1)
    A_1 = np.concatenate((np.ones([1, g]), A_1))
    A_2 = sigmoid(Theta2@A_1)
    t = (np.argmax(A_2, axis=0) + 1).T
    p = np.mean(t == y) * 100
    return p

input_layer_size = 400
hidden_layer_size = 25
num_layers = 10

mat = matio.loadmat("ex3data1.mat")
X = np.matrix(mat["X"])
y = np.matrix(mat["y"])

file = matio.loadmat("ex3weights.mat")
Theta1 = np.matrix(file["Theta1"])
Theta2 = np.matrix(file["Theta2"])

print("Training Set Accuracy:", predict(Theta1, Theta2, X, y))