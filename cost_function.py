import numpy as np


def compute_cost_function(x_matrix, y_matrix, number_of_iterations, alpha=0.01):

    """ This function helps you compute Cost Function(J) for on problems related to linear regression.
     Please note that this function does not care about the number of features in the problem.
     Outputs are : cost_function, theta_matrix, cost_function_list, theta_matrix_dict.
     For better understand of this code you must be familiar with cost function, gradient descent and how to
     crate a loop to find the best theta for the problem"""

    # Step 1: Defines variables--->
    # a) Set the first column of the x_matrix to ones to consider values for x(0)
    number_of_features = x_matrix.shape[1]
    number_of_training_examples = x_matrix.shape[0]
    matrix_ones = np.ones((number_of_training_examples, 1), dtype=int)
    x_matrix = np.append(matrix_ones, x_matrix, axis=1)

    # b) create a dictionary and a list for theta_matrix and cost_function to store all attained values during the loop
    theta_matrix_dict = {}
    cost_function_list = []

    # c) fill theta_matrix_dict with a list[0]for each theta to be able to have access to theta0_list,theta1_list and...
    # d) We put [0] because  the initial values for our thetas must be set to 0
    # e) we have (number_of_features + 1) thetas since we have theta0.
    for i in range(number_of_features+1):
        theta_matrix_dict[f'theta({i})'] = [0]

    # Step 2:set the initial values to thetas and determine the first value for cost function with thetas which are zero
    # As you know the formulas for our problem are:
    #      a)hypothesis = (theta(0) * x(0)) + (theta(1) * x(1)) + ...-->short way--->
    #      hypothesis_matrix = x_matrix * theta_matrix
    #      b)error = (hypothesis_matrix - y_matrix)
    #      C)J( cost function ) = (1/2*m)* sum( error^2 )
    theta_matrix = np.zeros((x_matrix.shape[1], 1))
    h_matrix = x_matrix.dot(theta_matrix)
    error = h_matrix - y_matrix
    s = np.sum(np.square(error))
    cost_function = (1 / (2 * number_of_training_examples)) * s
    cost_function_list.append(cost_function)
    print(f'The first values for thetas  before using gradient descent loop are:\n {theta_matrix}')
    print(f'\nThe first value  for cost_function  using gradient descent loop is :\n {cost_function}')
    print('----------------------------------------------------------------')

    # Step3: make a loop to find the minimum value for cost function with the help of Gradient Descent
    for i in range(number_of_iterations):

        # a: Update thetas with learning rate(alpha) to find the best thetas:

        #                    theta(0) = theta(0) - (alpha/m) * sum( error * x(0) )
        #                    theta(1) = theta(1) - (alpha/m) * sum( error * x(1) )
        #                                        .....
        #                    theta(p) = theta(p) - (alpha/m) * sum( error * x(p) )

        for p in range(number_of_features+1):
            theta_matrix[p, 0] = theta_matrix[p, 0] - ((alpha / number_of_training_examples) *
                                                       sum(error * np.transpose(x_matrix[:, p][np.newaxis])))
            theta_matrix_dict[f'theta({p})'].append(theta_matrix[p, 0])

        # b)Calculate the formulas with updated thetas
        h_matrix = x_matrix.dot(theta_matrix)
        error = h_matrix - y_matrix
        s = np.sum(np.square(error))
        cost_function = (1 / (2 * number_of_training_examples)) * s
        cost_function_list.append(cost_function)
    print(f'Optimized values for thetas are:\n {theta_matrix}')
    print(f'\nOptimized value  for cost_function is :\n {cost_function}')

    #  Step 4: Return the last J as cost_function,the best thetas as theta_matrix, list of J values
    #  as cost_function_list and the dictionary of  theta_matrix as theta_matrix_dict
    return cost_function, theta_matrix, cost_function_list, theta_matrix_dict
