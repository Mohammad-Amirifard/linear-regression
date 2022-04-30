import numpy as np
from matplotlib import pyplot as plt


def plot_fitline(x, y, theta_matrix):
    """ This function helps you draw fit-line graph for data
    Output is a graph"""

    # Step1: Set the figure structure
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('Our Graphs', fontweight='bold', fontsize=20)

    # Step2: Set properties for the left plot
    axs[0].scatter(x=x, y=y, c='blue', marker='*', linewidths=4, label='Data')
    axs[0].grid()
    axs[0].set(xlabel='Population in 10,000s', ylabel='Price in $10,000s')
    axs[0].legend()

    # Step3: Set properties for the right plot
    axs[1].scatter(x=x, y=y, c='blue', marker='*', linewidths=4, label='Data')

    # Step4: Draw fit line: for fit line we need to use
    # hypothesis = (theta(0) * x(0)) + (theta(1) * x(1)) + ...---> or short way which is
    # hypothesis_matrix = x_matrix * theta_matrix
    # So we must Set the first column of the x_matrix to ones to consider values for x(0) and create xmatrix like before
    number_of_training_examples = x.shape[0]
    matrix_ones = np.ones((number_of_training_examples, 1), dtype=int)
    x_matrix = np.concatenate((matrix_ones, x), axis=1)
    h = x_matrix.dot(theta_matrix)
    axs[1].plot(x, h,
                label=f'Fit line => h ={"{:.3f}".format(theta_matrix[0,0])}+'
                      f'{"{:.3f}".format(theta_matrix[1,0])}*x', color='red')
    axs[1].grid()
    axs[1].set(xlabel='Population in 10,000s', ylabel='Price in $10,000s')
    axs[1].legend()


def plot_costfunction_trend(number_of_iteration, cost_function_list):
    """ This function helps you draw cost function_trend graph.
     Output is the trend showing the direction of cost_function"""

    # Step1:Set the figure structure
    figure = plt.figure(figsize=(8, 6))

    # Step2: plot the line chart for cost function trend using cost_function_list we got in the previous function
    plt.plot(np.arange(0, number_of_iteration+1), cost_function_list, linewidth=3,
             linestyle='dashed', label='Behaviour of the J')

    # Step3: Set Start and End points to show the initial and last values for Cost Function
    plt.scatter(x=0, y=cost_function_list[0], c='red', marker='v', linewidths=4, label='Start Point')
    plt.scatter(x=number_of_iteration, y=cost_function_list[-1], c='black', marker='>',
                linewidths=4, label='End Point')
    plt.xlabel('Number of iteration', fontweight='bold', fontsize=14)
    plt.ylabel('Cost Function (J)', fontweight='bold', fontsize=14)
    plt.grid()
    plt.legend()


def plot_costfunction_and_contour2d(x, y, theta_matrix, cost_function):
    """ This function helps you draw cost function graph.
     Output includes two graphs: 1st is the 3d graph , 2nd is a 2d contour"""

    # Draw the first graph: 3d graph
    # Step1: Set optional line space for theta0 and theta1 to have view of data
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # Step2: create a matrix of zeros with len(theta0)*len(theta1) dimensions and call it j_vals
    j_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    # Step3: Fill the above matrix with new j in which we must use x_matrix,
    # y_matrix and new theta0_vals and theta1_vals
    for i in range(len(theta0_vals)):
        for p in range(len(theta1_vals)):
            t = np.array([[theta0_vals[i]], [theta1_vals[p]]])
            number_of_training_examples = x.shape[0]
            matrix_ones = np.ones((number_of_training_examples, 1), dtype=int)
            x_matrix = np.concatenate((matrix_ones, x), axis=1)
            h_matrix = x_matrix.dot(t)
            error = h_matrix - y
            s = np.sum(np.square(error))
            j = (1 / (2 * number_of_training_examples)) * s
            j_vals[i, p] = j

    # Step4: So far we created a Cost function matrix with len(theta0)*len(theta1) dimensions
    # and now, It's time to draw it with the help of contour3D plot
    j_vals = np.transpose(j_vals)

    # a) Set properties to plot contour3D
    theta0, theta1 = np.meshgrid(theta0_vals, theta1_vals)
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')
    b = ax.contour3D(theta0, theta1, j_vals, levels=1000)
    ax.set_xlabel('Theta0', weight='bold', size=12)
    ax.set_ylabel('Theta1', weight='bold', size=12)
    ax.set_zlabel('Cost Function ', weight='bold', size=10)
    ax.set_title('3D contour of Cost Function', weight='bold', size=20)

    # b) Set properties to show the last point of cost function which we could determine from gradient descent
    ax.scatter3D(theta_matrix[0, 0], theta_matrix[1, 0], cost_function, marker='*', c='red', linewidth=10)
    fig.colorbar(b, shrink=2.5, aspect=3.5, label='colorbar')
    ax.view_init(5, 220)

    # Draw the second graph: contour2d
    # Step1: Set properties to plot contour2D
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes()
    b = ax.contour(theta0, theta1, j_vals, levels=100)
    ax.set_xlabel('Theta0', weight='bold', size=12)
    ax.set_ylabel('Theta1', weight='bold', size=12)
    ax.set_title('2D contour of Cost Function', weight='bold', size=20)

    # Step2: Set properties to show the End Point in which the Minimum of J occurs
    ax.scatter(0, 0, marker='*', c='black', linewidth=6, label='Start point')
    ax.scatter(theta_matrix[0], theta_matrix[1], marker='*', c='red', linewidth=6,
               label='End Point in which the Minimum of J occurs ')
    ax.legend()