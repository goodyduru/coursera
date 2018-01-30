import scipy.io as matio
import scipy.optimize as optimize
import numpy as np


def sigmoid(z):
    g = np.divide(1, (1 + np.exp(-z)))
    return g

def compute_cost_regularized(theta, X, y, lda):
    reg =lda/(2*len(y)) * np.sum(theta[1:]**2)
    return 1/len(y) * np.sum(-y @ np.log(sigmoid(X@theta))
                             - (1-y) @ np.log(1-sigmoid(X@theta))) + reg


def compute_gradient_regularized(theta, X, y, lda):
    XT = X.T
    beta = sigmoid(X@theta) - y
    regterm = lda/len(y) * theta
    # theta_0 does not get regularized, so a 0 is substituted in its place
    regterm[0] = 0
    gradient = (1/len(y) * XT@beta).T + regterm
    return np.squeeze(np.asarray(gradient))


def costFunction(theta, X, y, reg):
    (m, n) = np.shape(X)
    theta = theta.reshape((n, 1))
    grad = np.zeros(np.shape(theta))
    my_x = sigmoid((X@theta).T)
    term1 = np.log(my_x)
    term2 = np.log(1 - my_x)
    regu = (reg/(2 * m)) * np.sum( np.power(theta[1:n], 2) )
    J =  ( ((term1@-y) - (term2@(1 - y))) / m ) + regu
    return J

def Gradient(theta, X, y, reg):
    (m, n) = np.shape(X)
    theta = theta.reshape((n, 1))
    my_x = sigmoid((X@theta).T)
    grad = (((my_x - np.transpose(y))@X).T) / m + (reg * theta/m)
    grad[0] = grad[0] - (reg * theta[0]/m)
    return np.squeeze(np.asarray(grad))


def oneVsAll(X, y, num_labels, reg):
    (m, n) = np.shape(X)
    all_theta = np.zeros(([num_labels, n + 1]))
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    c = 1
    while c <= num_labels:
        initial_theta = np.zeros([n+1, 1])
        result = optimize.minimize(costFunction, initial_theta, args=(X, np.where((y == c), 1, 0), reg), method='CG', jac=Gradient, options={'disp': True, 'maxiter': 100})
        #result = optimize.fmin_tnc(costFunction, initial_theta, args = (X, np.where((y == c), 1, 0), reg), fprime=Gradient)
        all_theta[c - 1, :] = np.matrix(result.x)
        c += 1
    return all_theta


def predictOneVsAll(all_theta, X, y):
    (m, n) = np.shape(X)
    X = np.concatenate((np.ones([m, 1]), X), axis=1)
    return np.mean(np.argmax(sigmoid(X@all_theta.T), axis=1)+1 == y)*100

input_layer_size = 400
num_labels = 10

mat = matio.loadmat("ex3data1.mat")
X = np.matrix(mat['X'])
y = np.matrix(mat['y'])

print("Testing logistic regression with regularisation")

theta_t = np.matrix("-2; -1; 1; 2")
X_t = np.concatenate((np.ones([5, 1]), np.arange(1, 16).reshape((5, 3), order='F') / 10), axis=1)
y_t = np.matrix("1; 0; 1; 0; 1")
lambda_t = 3
J = costFunction(theta_t, X_t, y_t, lambda_t)
grad = Gradient(theta_t, X_t, y_t, lambda_t)
print(np.shape(grad))
print("Cost", J)
print("Gradient", grad)

reg = 0.1
all_theta = oneVsAll(X, y, num_labels, reg)
predict = predictOneVsAll(all_theta, X, y)
print("Training Set Accuracy", predict)
