import numpy as np
import scipy.io as matio
import scipy.optimize as optimize
import matplotlib.pyplot as pyplot


def cost_function(input_array, output_matrix, ratings, users, movies, features, lbd):
    input_matrix = np.reshape(input_array[0:movies*features], (movies, features))
    weight = np.reshape(input_array[movies*features:], (users, features))
    error = np.power((np.dot(input_matrix, weight.T) - output_matrix), 2)
    cost = np.sum(np.multiply(ratings, error))/2 + ((lbd/2) * np.sum(np.power(weight, 2))) + \
           ((lbd/2) * np.sum(np.power(input_matrix, 2)))
    return cost


def gradient_function(input_array, output_matrix, ratings, users, movies, features, lbd):
    input_matrix = np.reshape(input_array[0:movies*features], (movies, features))
    weight = np.reshape(input_array[movies*features:], (users, features))
    num_rows, num_cols = ratings.shape
    current_row, current_col = 0, 0
    input_gradient, weight_gradient = np.zeros(input_matrix.shape), np.zeros(weight.shape)

    while current_row < num_rows:
        rating_col = np.ravel(ratings[current_row, :])
        indexes = np.where((rating_col == 1))
        temp_weight = weight[indexes[0], :]
        temp_output = output_matrix[current_row, indexes[0]]
        input_gradient[current_row, :] = (input_matrix[current_row, :] @ temp_weight.T - temp_output) @ temp_weight
        current_row += 1

    input_gradient = input_gradient + (lbd * input_matrix)
    while current_col < num_cols:
        rating_row = np.ravel(ratings[:, current_col])
        indexes = np.where((rating_row == 1))
        temp_input = input_matrix[indexes[0], :]
        temp_output = output_matrix[indexes[0], current_col]
        weight_gradient[current_col, :] = (np.matrix(temp_input @ weight[current_col, :].T) - temp_output.T) @ temp_input
        current_col += 1

    weight_gradient = weight_gradient + (lbd * weight)
    gradients = np.concatenate((np.ravel(input_gradient), np.ravel(weight_gradient)))
    return gradients


def check_cost_function(lbd):
    input_matrix_temp = np.random.rand(4, 3)
    weight_temp = np.random.rand(5, 3)
    output_matrix = input_matrix_temp @ weight_temp.T
    random = np.random.rand(output_matrix.shape[0], output_matrix.shape[1])
    is_greater_than_half = np.where((random > 0.5))
    output_matrix[is_greater_than_half[0], is_greater_than_half[1]] = 0
    ratings = np.where((output_matrix != 0), 1, 0)

    input_matrix = np.random.randn(input_matrix_temp.shape[0], input_matrix_temp.shape[1])
    weight = np.random.randn(weight_temp.shape[0], weight_temp.shape[1])
    users = output_matrix.shape[1]
    movies = output_matrix.shape[0]
    features = weight_temp.shape[1]

    args = [output_matrix, ratings, users, movies, features, lbd]
    cost_func = cost_function
    input_array = np.concatenate((np.ravel(input_matrix), np.ravel(weight)))
    num_grad = compute_numerical_gradient(cost_func, input_array, args)
    grad = gradient_function(input_array, output_matrix, ratings, users, movies, features, lbd)
    print(num_grad, grad)
    print("The above two should be very similar\n\n")
    diff = np.linalg.norm(num_grad - grad)/np.linalg.norm(num_grad + grad)
    print("Difference ", diff)


def compute_numerical_gradient(cost_func, input_array, args):
    num_grad = np.zeros(np.size(input_array))
    perturb = np.zeros(np.size(input_array))

    e = 1E-4
    p = 0
    while p < np.size(input_array):
        perturb[p] = e
        loss1 = cost_func(input_array - perturb, *args)
        loss2 = cost_func(input_array + perturb, *args)

        num_grad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
        p += 1
    return num_grad


def load_movie_list():
    movies = list()
    with open('movie_ids.txt') as movie_file:
        cell = movie_file.readlines()
    for line in cell:
        split = line.split()
        split = split[1:]
        join = " ".join(split)
        movies.append(join)
    return movies


def normalize_ratings(output_matrix, ratings):
    num_rows, num_cols = output_matrix.shape
    output_mean = np.zeros((num_rows, 1))
    output_norm = np.zeros((num_rows, num_cols))
    pos = 0
    while pos < num_rows:
        indexes = np.where((ratings[pos, :] == 1))
        output_mean[pos] = np.mean(output_matrix[pos, indexes[1]])
        output_norm[pos, indexes[0]] = output_matrix[pos, indexes[1]] - output_mean[pos]
        pos += 1

    return output_mean, output_norm

file = matio.loadmat('ex8_movies.mat')
# User and ratings
Y = np.matrix(file["Y"])

# Conditions if User rates a movie
R = np.matrix(file["R"])


file = matio.loadmat("ex8_movieParams.mat")
X = np.matrix(file["X"])
Theta = np.matrix(file["Theta"])

col = np.ravel(R[0, :])
tec = np.where((col == 1))
mean = np.mean(Y[0, tec[0]])
print(mean)

print("Plotting The Recommendations", end="\n")
pyplot.imshow(Y)
pyplot.xlabel("Users")
pyplot.ylabel("Movies")
pyplot.show()

num_users, num_movies, num_features = 4, 5, 3
X = X[0:num_movies, 0:num_features]
Theta = Theta[0:num_users, 0:num_features]
Y = Y[0:num_movies, 0:num_users]
R = R[0:num_movies, 0:num_users]

params = np.concatenate((np.ravel(X), np.ravel(Theta)))
J = cost_function(params, Y, R, num_users, num_movies, num_features, 0)
print(J)

print("Checking gradients without regularization\n")
check_cost_function(0)

J = cost_function(params, Y, R, num_users, num_movies, num_features, 1.5)
print(J)

print("Checking gradients with regularization\n")
check_cost_function(1.5)

movie_list = load_movie_list()
my_ratings = np.zeros((1682, 1))
my_ratings[0] = 4
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5
print("\nNew Users Ratings\n")
length = len(my_ratings)
i = 0

while i < length:
    if my_ratings[i] > 0:
        print("Rated ", my_ratings[i], "for movie", movie_list[i])
    i += 1

print("Training Collaborative Filtering")
file = matio.loadmat('ex8_movies.mat')
# User and ratings
Y = np.matrix(file["Y"])

# Conditions if User rates a movie
R = np.matrix(file["R"])


Y = np.concatenate((my_ratings, Y), axis=1)
where = np.where((my_ratings != 0), 1, 0)
R = np.concatenate((where, R), axis=1)
Y_mean, Y_norm = normalize_ratings(Y, R)
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)
initial_params = np.concatenate((np.ravel(X), np.ravel(Theta)))
reg = 10

theta = optimize.fmin_cg(cost_function, initial_params, args=(Y_norm, R, num_users, num_movies, num_features, reg),
                         fprime=gradient_function, maxiter=100)
print("Recommender system training done\n")
X = np.reshape(theta[0:num_movies*num_features], (num_movies, num_features))
Theta = np.reshape(theta[num_movies*num_features:], (num_users, num_features))
prod = X @ Theta.T
my_predictions = np.transpose(np.matrix(prod[:, 0])) + Y_mean
movie_list = load_movie_list()
sort_index = np.argsort(my_predictions, axis=0)
last = np.ravel(sort_index[len(sort_index) - 10:])
my_predictions = np.ravel(my_predictions)
i = 0
while i < 10:
    j = last[i]
    print("Predicted ", my_predictions[j], "for movie ", movie_list[j], end="\n")
    i += 1
print("\nOriginal ratings provided\n")
while i < length:
    if my_ratings[i] > 0:
        print("Rated ", my_ratings[i], "for movie", movie_list[i])
    i += 1
