import scipy.io as matio
import scipy.optimize as optimize
import numpy as np


def sigmoid(z):
    g = np.divide(1, (1 + np.exp(-z)))
    return g


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


def rand_initialize_weights(l_in, l_out):
    epsilon_init = 0.12
    w = np.random.rand(l_out, 1 + l_in) * 2 * epsilon_init - epsilon_init
    return np.matrix(w)


def debug_initialize_weights(l_out, l_in):
    w = np.zeros([l_out, 1 + l_in])
    w = np.reshape(np.sin(np.arange(1, 1 + np.size(w))), np.shape(w), order='F') / 10
    return np.matrix(w)


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbd):
    nn_params = np.matrix(nn_params);
    # Get a 25 by 401 matrix
    theta1 = np.reshape(nn_params[:, 0:(hidden_layer_size * (input_layer_size + 1))], [hidden_layer_size, input_layer_size + 1])

    # Get a 10 by 26 matrix
    theta2 = np.reshape(nn_params[:, (hidden_layer_size * (input_layer_size + 1)):], [num_labels, hidden_layer_size + 1])

    # Number of rows of X
    m = len(X)
    # convert y to a 5000 by 25 matrix
    new_y = np.zeros([m, num_labels])
    k = 0
    while k < m:
        label = np.zeros([num_labels, 1])
        t = y[k]
        label[t-1] = 1
        new_y[k, :] = label.flatten()
        k += 1

    # add one column to X
    X = np.concatenate((np.ones([m, 1]), X), axis=1)
    # z2 is a 25 by 5000 matrix
    z2 = theta1 @ X.T
    # a_2 is the sigmoid of z2, it is a 25 by 5000 matrix
    a2 = sigmoid(z2)
    # add one row to a2 to make it a 26 by 5000 matrix
    a2 = np.concatenate((np.ones([1, m]), a2))
    # z3 is a 10 by 5000 matrix
    z3 = theta2 @ a2
    # a3 is the sigmoid of z3
    a3 = sigmoid(z3)

    regularization = (lbd / (2 * m)) * (np.sum(np.sum(np.power(theta1[:, 1:], 2))) + np.sum(np.sum(np.power(theta2[:, 1:], 2))))
    cost = (np.sum(np.sum(np.multiply(np.log(a3.T), -new_y ) - np.multiply(np.log(1 - a3.T), (1 - new_y))))) / m + regularization

    return cost


def nn_gradients(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbd):
    nn_params = np.matrix(nn_params)
    # Get a 25 by 401 matrix
    theta1 = np.reshape(nn_params[:, 0:(hidden_layer_size * (input_layer_size + 1))], [hidden_layer_size, input_layer_size + 1])

    # Get a 10 by 26 matrix
    theta2 = np.reshape(nn_params[:, (hidden_layer_size * (input_layer_size + 1)):], [num_labels, hidden_layer_size + 1])

    # Number of rows of X
    m = len(X)

    theta1_grad = np.zeros(np.shape(theta1))
    theta2_grad = np.zeros(np.shape(theta2))
    # convert y to a 5000 by 25 matrix
    new_y = np.zeros([m, num_labels])
    k = 0
    while k < m:
        label = np.zeros([num_labels, 1])
        t = y[k]
        label[t-1] = 1
        new_y[k, :] = label.flatten()
        k += 1

    # add one column to X
    X = np.concatenate((np.ones([m, 1]), X), axis=1)
    # z2 is a 25 by 5000 matrix
    z2 = theta1 @ X.T
    # add one row to the sigmoid gradient of z2, therefore sigmoid_z2 becomes a 26 by 5000 matrix
    sigmoid_z2 = np.concatenate((np.ones([1, m]), sigmoid_gradient(z2)))
    # a_2 is the sigmoid of z2, it is a 25 by 5000 matrix
    a2 = sigmoid(z2)
    # add one row to a2 to make it a 26 by 5000 matrix
    a2 = np.concatenate((np.ones([1, m]), a2))
    # z3 is a 10 by 5000 matrix
    z3 = theta2 @ a2
    # a3 is the sigmoid of z3
    a3 = sigmoid(z3)
    """
       Start of Backpropagation
       Start by converting a2 and a3 to their transpose i.e
       a3 becomes a 5000 by 10 matrix
       a2 becomes a 5000 by 26 matrix
   """
    a3 = a3.T
    a2 = a2.T
    # delta3 is a 10 by 5000 matrix
    delta3 = (a3 - new_y).T
    # delta2 is a 26 by 5000 matrix
    delta2 = np.multiply((theta2.T @ delta3), sigmoid_z2)
    # intermediate is a 25 by 401 matrix
    intermediate = delta2[1:, :] @ X
    # this is a 25 by 401 matrix
    theta1_grad += intermediate
    # this ia a 10 by 26 matrix
    theta2_grad += delta3 @ a2
    theta1_grad[:, 0] = theta1_grad[:, 0] / m
    theta1_grad[:, 1:] = ( theta1_grad[:, 1:] / m ) + (lbd / m * theta1[:, 1:])
    theta2_grad[:, 0] = theta2_grad[:, 0] / m
    theta2_grad[:, 1:] = ( theta2_grad[:, 1:] / m ) + (lbd / m * theta2[:, 1:])
    grad = np.concatenate((theta1_grad.ravel(), theta2_grad.ravel()))
    return grad


def compute_numerical_gradients(costFun, theta, args):
    numgrad = np.zeros(np.size(theta))
    perturb = np.zeros(np.size(theta))

    e = 1E-4
    p = 0
    while p < np.size(theta):
        perturb[p] = e
        loss1 = costFun(theta - perturb, *args)
        loss2 = costFun(theta + perturb, *args)

        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
        p += 1
    return numgrad



def check_nn_gradients(lbd):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    theta2 = debug_initialize_weights(num_labels, hidden_layer_size)

    X = debug_initialize_weights(m, input_layer_size - 1)
    y = np.transpose(np.matrix(1 + np.mod(np.arange(1, 6), num_labels)))
    nn_params = np.concatenate((theta1.ravel(), theta2.ravel()), axis=1)
    grad = nn_gradients(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbd)
    costFunc = nn_cost_function
    args = [input_layer_size, hidden_layer_size, num_labels, X, y, lbd]
    numgrad = compute_numerical_gradients(costFunc, nn_params, args)
    print(grad, numgrad)

    diff = np.linalg.norm(numgrad - grad)/np.linalg.norm(numgrad + grad)
    print("Difference ", diff)


def predict(theta1, theta2, X, y):
    m = len(X)
    num_labels = len(theta1)

    p = np.zeros([m, 1])
    h1 = sigmoid(np.concatenate((np.ones([m, 1]), X), axis=1) @ theta1.T)
    h2 = sigmoid(np.concatenate((np.ones([m, 1]), h1), axis=1) @ theta2.T)
    p = h2.argmax(1) + 1
    return np.mean(p == y) * 100


# initialize all layer sizes
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

# load matrix file
file = matio.loadmat("ex4data1.mat")
X = np.matrix(file["X"])
y = np.matrix(file["y"])
m = len(X)

# load weights file
mat = matio.loadmat("ex4weights.mat")
Theta1 = np.matrix(mat["Theta1"])
Theta2 = np.matrix(mat["Theta2"])
#flatten all the weights
nn_params = np.concatenate((Theta1.ravel(), Theta2.ravel()), axis=1)
j = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 0)
print("Cost at lambda = 0", j)
j = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 1)
print("Cost at lambda = 1", j)
g = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
print("Sigmoid gradient", g)

initial_theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
initial_theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
initial_nn_params = np.concatenate((initial_theta1.ravel(), initial_theta2.ravel()), axis=1)
check_nn_gradients(0)
check_nn_gradients(3)
j = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 3)
print("Cost at lambda = 3", j)
#result = optimize.minimize(nn_cost_function, initial_nn_params, args=(input_layer_size, hidden_layer_size, num_labels, X, y, 1), method='CG', jac=nn_gradients, options={'disp': True, 'maxiter': 50})
result = optimize.fmin_cg(nn_cost_function, initial_nn_params, args=(input_layer_size, hidden_layer_size, num_labels, X, y, 1), fprime=nn_gradients, maxiter=100)
result_params = np.matrix(result)
theta1 = np.reshape(result_params[:, 0:(hidden_layer_size * (input_layer_size + 1))], [hidden_layer_size, input_layer_size + 1])

theta2 = np.reshape(result_params[:, (hidden_layer_size * (input_layer_size + 1)):], [num_labels, hidden_layer_size + 1])
print(predict(theta1, theta2, X, y))