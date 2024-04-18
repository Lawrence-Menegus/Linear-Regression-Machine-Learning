# Program 3 
# Lawrence Menegus 
# Machine Learning CPSC 429
# Program Description:  

import numpy as np
import matplotlib.pyplot as plt

# Arrarys for the error and ErrorDelta values 
errors = []
errors1 = []
errorDelta = []
errorDelta1 = []

# Normalize the values Function
def normalize_features(features):
    # Find the minimum value of each column
    min_val = np.min(features, axis=0)
    # Find the maximum value of each column
    max_val = np.max(features, axis=0)
    # Apply min-max scaling to each column
    normalized_features = -1 + 2 * (features - min_val) / (max_val - min_val)
    # Return the normalized features
    return normalized_features

# Define a function to compute the cost for linear regression
def compute_cost(X, y, theta):
    # Number of training samples
    m = y.size
    # Compute the predictions using the hypothesis function
    predictions = X.dot(theta)
    # Compute the squared errors between the predictions and the actual values
    errors = (y - predictions)
    # Compute cost function
    J = (1.0/2) * errors.T.dot(errors) 
    # Return the cost value
    return J

# Gradient Descent Algorithm
def gradient_descent(X, y, weights, alpha, iterations):
    m = y.size
    J_history = np.zeros(shape=(iterations, 1))

    for i in range(iterations):
        # Compute prediction and error
        prediction = np.dot(X, weights)
        error = prediction - y

        # Compute gradient
        gradient = np.dot(X.transpose(), error)

        # Update weights
        weights = weights - alpha * gradient

        # Compute cost of updated weights
        cost = compute_cost(X, y, weights)
        J_history[i, 0] = cost

    return weights, J_history

# Input 1 results for Error, Squared Error, Predictions, Error Delta w[0], w[1], w[2], w[3]
def input_1_results(X, y, weights):
    for i in range(len(y)):
        # target
        target = y[i]
        # Calculate the Prediction
        prediction = np.dot(X[i], weights)
        # Calculate the error
        error = y[i] - prediction
        # Error Squared Equations
        errorsqr = error ** 2
        # Calculate the Error Delta
        Delta = X[i] * error
        # Put target, prediction, error, and errorsqr in an array
        temp = [target, prediction, error, errorsqr]
        errors.append(temp)
        # Error Deltas for Input 1
        errorDeltas = [Delta[0], Delta[1], Delta[2], Delta[3]]
        errorDelta.append(errorDeltas)

# Input 2 Results for Error, Squared Error, Predictions, Error Delta w[0], w[1], w[2]
def input_2_results(X, y, weights):
    for i in range(len(y)):
        # target
        target = y[i]
        # Calculate the Prediction
        prediction = np.dot(X[i], weights)
        # Calculate the error
        error = y[i] - prediction
        # Error Squared Equations
        errorsqr = error ** 2
        # Calculate the Error Delta
        Delta = X[i] * error
        # Put target, prediction, error, and errorsqr in an array
        temp = [target, prediction, error, errorsqr]
        errors1.append(temp)
        # Error Deltas for Input 2
        errorDeltas = [Delta[0], Delta[1], Delta[2]]
        errorDelta1.append(errorDeltas)

# Input 1 method
def input_1():
    print('\nData File: prog3_input1.txt')
    data = np.genfromtxt('prog3_input1.txt', delimiter=',')

    # Place values into an array
    X = data[:, 1:4]
    y = data[:, 5]

    X = np.c_[np.ones(X.shape[0]), X]

    # Initial Weights
    weights = [-0.146, 0.185, -0.044, 0.119]

    # Clears previous errors and Error Deltas
    global errors, errorDelta
    errors.clear()
    errorDelta.clear()

    # Printout for Initial Iteration
    print('\nInitial weights:')
    print([weights], '\n')
    input_1_results(X, y, weights)

    # Print out the Errors
    print('\nErrors:')
    b_error = np.array(errors)
    print(b_error)

    # Print out the Error Deltas
    print('\nErrorDelta:')
    b_delta = np.array(errorDelta)
    print(b_delta)

    # Alpha for the new weights
    alpha = 0.00000002

    # Printout for Iteration 1
    iterations = 1
    new_weights, cost_history = gradient_descent(X, y, weights, alpha, iterations)

    # Clears previous errors and Error Deltas
    errors.clear()
    errorDelta.clear()

    # Print out the New weights
    print('\nNew weights after iteration 1')
    print(np.array([new_weights]), "\n")
    input_1_results(X, y, new_weights)

    # Print out the Errors
    print('\nErrors:')
    n_errors = np.array(errors)
    print(n_errors)

    # Print out the error Deltas
    print('\nErrorDelta:')
    n_delta = np.array(errorDelta)
    print(n_delta)

    # Printout for Iteration 2 new weights
    iterations = 2
    new_weights, cost_history = gradient_descent(X, y, weights, alpha, iterations)
    print('\nNew weights after iteration 2')
    print(np.array([new_weights]), "\n")

    # Printout for Final iteration of 100
    iterations = 100
    new_weights, cost_history = gradient_descent(X, y, weights, alpha, iterations)
    print('Final weights after 100 iterations:')
    print(np.array([new_weights]), "\n")

    # Calculate the Error Squared Sums
    errsqr_sum = compute_cost(X, y, new_weights)
    print('Final Sum of squared errors:', f"[{errsqr_sum:.7f}]")

    # Graph for the Cost function and iterations plots
    plt.figure()
    plt.plot(range(1, len(cost_history) + 1), cost_history, color='blue', linestyle='-')
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.show()


# Input 2 Method
def input_2():
    print('\n\n\nData File: prog3_input2.txt')
    data = np.genfromtxt('prog3_input2.txt', delimiter=',')

    # Place values into an array
    X = data[:, 2:]
    y = data[:, 1]

    X = np.c_[np.ones(X.shape[0]), X]

    weights = [-59.50, -0.15, 0.60]

    # Clears previous errors and Error Deltas
    global errors1, errorDelta1
    errors1.clear()
    errorDelta1.clear()

    # Printout for Initial Iteration
    print('\nInitial weights:')
    print([weights], '\n')
    input_2_results(X, y, weights)

    # Print out the Errors
    print('\nErrors:')
    v_error = np.array(errors1)
    print(v_error)

    # Print out the Error Deltas
    print('\nErrorDelta')
    v_delta = np.array(errorDelta1)
    print(v_delta)

    # Alpha for the new weight/ Gradient Descent
    alpha = 0.000002

    # Printout for Iteration 1
    iterations = 1
    new_weights, cost_history = gradient_descent(X, y, weights, alpha, iterations)

    # Clears previous errors and Error Deltas
    errors1.clear()
    errorDelta1.clear()

    print('\nNew weights after iteration 1')
    print(np.array([new_weights]))
    input_2_results(X, y, new_weights)

    # Print out the Errors
    print('\nErrors:')
    c_error = np.array(errors1)
    print(c_error)

    # Print out the error Deltas
    print('\nErrorDelta')
    c_delta = np.array(errorDelta1)
    print(c_delta)

    # Printout for Iteration 2 new weights
    iterations = 2
    new_weights, cost_history = gradient_descent(X, y, weights, alpha, iterations)
    print('\nNew weights after iteration 2')
    print(np.array([new_weights]), "\n")

    # Printout for Final iteration of 100
    iterations = 100
    new_weights, cost_history = gradient_descent(X, y, weights, alpha, iterations)
    print('Final weights after 100 iterations:')
    print(np.array([new_weights]), "\n")

    # Calculate the Error Squared Sums
    errsqr_sum = compute_cost(X, y, new_weights)
    print('Final Sum of squared errors:', f"[{errsqr_sum:.8f}]")

    # Graph for the Cost function and iterations plots
    plt.figure()
    plt.plot(range(1, len(cost_history) + 1), cost_history, color='blue', linestyle='-')
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.show()

# Calls the functions
input_1()
input_2()
