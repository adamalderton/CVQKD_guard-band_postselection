"""
This a file for working out the bugs/details of recovering the correct marginals and other numerical integrations from
the 4D distribution of data.
"""
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

grid_range = [-5, 5]
num_points_on_axis = 64
x = np.linspace(grid_range[0], grid_range[1], num_points_on_axis)

def generate_data_2D(cov_matrix):
    """
        Generates a 2D gaussian distribution with the given covariance matrix, centred at the origin.
        :param cov_matrix: 2x2 covariance matrix
        :return: 2D gaussian distribution
    """
    x_mesh, y_mesh = np.meshgrid(x, x)

    return multivariate_normal(np.zeros(2), cov_matrix).pdf(np.dstack((x_mesh, y_mesh)))

def calculate_2D_marginals(data):
    """
        Calculates the marginal distributions of a 2D data distribution.
        :param data: 2D numpy array representing the data
        :return: 1D numpy arrays representing the marginal distributions along the X and Y axes
    """
    x_marginal = np.sum(data, axis=0)
    y_marginal = np.sum(data, axis=1)

    return x_marginal, y_marginal

def plot_2D_data(data, x_marginal, y_marginal):
    """
        Plots the 2D data in a heatmap and the marginal distributions in two additional subplots.
        :param data: 2D numpy array representing the data
    """
    # Create the figure and axis objects
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the heatmap of the 2D data
    cax = ax1.imshow(data, extent=(grid_range[0], grid_range[1], grid_range[0], grid_range[1]), 
                    origin='lower', cmap='viridis', aspect='auto')
    
    # Add a colorbar to indicate the scale of the heatmap
    fig.colorbar(cax, ax=ax1)

    # Set labels and titles for the heatmap
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_title('2D Gaussian Distribution')

    ax2.plot(x, x_marginal)
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Probability')
    ax2.set_title('Marginal Distribution (X axis)')

    ax3.plot(x, y_marginal)
    ax3.set_xlabel('Y axis')
    ax3.set_ylabel('Probability')
    ax3.set_title('Marginal Distribution (Y axis)')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

def calculate_2D_marginal_properties(marginal):

    marginal_integrated = np.trapz(y = marginal, x = x)

    mean = np.trapz(y = marginal * x, x = x) / marginal_integrated

    variance = np.trapz(y = marginal * (x - mean)**2, x = x) / marginal_integrated

    return marginal_integrated, mean, variance

#------------------------------------

def generate_data_4D(cov_matrix):
    """
        Generates a 4D gaussian distribution with the given covariance matrix, centred at the origin.
        :param cov_matrix: 4x4 covariance matrix
        :return: 4D gaussian distribution
    """
    
    x_mesh, y_mesh, z_mesh, w_mesh = np.meshgrid(x, x, x, x, indexing='ij')
    
    # Reshape the meshgrid arrays to be able to stack them correctly
    pos = np.stack([x_mesh, y_mesh, z_mesh, w_mesh], axis=-1)
    
    # Flatten the last dimension to create a 2D array where each row is a 4D point
    pos_flat = pos.reshape(-1, 4)
    
    # Calculate the PDF values for each 4D point
    pdf_values = multivariate_normal(np.zeros(4), cov_matrix).pdf(pos_flat)
    
    # Reshape the PDF values back to the original 4D grid shape
    data = pdf_values.reshape(num_points_on_axis, num_points_on_axis, num_points_on_axis, num_points_on_axis)
    
    return data

def calculate_4D_marginals(data):
    """
    Calculates the marginal distributions of a 4D data distribution.
    :param data: 4D numpy array representing the data
    :param x: 1D numpy array representing the axis values
    :return: A dictionary with the marginal distributions, means, and variances for each axis
    """
    # Calculate the marginal distributions along each axis
    x_marginal = np.sum(data, axis=(1, 2, 3))
    y_marginal = np.sum(data, axis=(0, 2, 3))
    z_marginal = np.sum(data, axis=(0, 1, 3))
    w_marginal = np.sum(data, axis=(0, 1, 2))

    # Store marginals in a dictionary
    marginals = {
        'x': {"marginal": x_marginal, "mean": None, "variance": None},
        'y': {"marginal": y_marginal, "mean": None, "variance": None},
        'z': {"marginal": z_marginal, "mean": None, "variance": None},
        'w': {"marginal": w_marginal, "mean": None, "variance": None}
    }

    # Normalize each marginal distribution
    for key in marginals:
        marginal = marginals[key]["marginal"]
        normalization = np.sum(marginal)
        normalized_marginal = marginal / normalization
        marginals[key]["marginal"] = normalized_marginal

        # Calculate the mean
        marginals[key]["mean"] = np.sum(normalized_marginal * x)

        # Calculate the variance
        marginals[key]["variance"] = np.sum(normalized_marginal * (x - marginals[key]["mean"])**2)

    return marginals

def plot_4D_data(x_marginal, y_marginal, z_marginal, w_marginal):
    """
        Plots the 4D data in four subplots, one for each marginal distribution.
        :param data: 4D numpy array representing the data
    """
    # Create the figure and axis objects
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot the marginal distributions along each axis
    ax1.plot(x, x_marginal)
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Probability')
    ax1.set_title('Marginal Distribution (X axis)')
    
    ax2.plot(x, y_marginal)
    ax2.set_xlabel('Y axis')
    ax2.set_ylabel('Probability')
    ax2.set_title('Marginal Distribution (Y axis)')
    
    ax3.plot(x, z_marginal)
    ax3.set_xlabel('Z axis')
    ax3.set_ylabel('Probability')
    ax3.set_title('Marginal Distribution (Z axis)')
    
    ax4.plot(x, w_marginal)
    ax4.set_xlabel('W axis')
    ax4.set_ylabel('Probability')
    ax4.set_title('Marginal Distribution (W axis)')
    
    # Adjust the spacing between subplots
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def calculate_4D_covariance_element(data, marginals, i, j):
    """
        Summary of Steps:
        1. Calculate the joint marginal distribution for variables Xi and Xj by integrating out the other variables from the 4D distribution.
        2. Calculate the individual marginal distributions for Xi and Xj by integrating the joint marginal distribution over the other variable.
        3. Calculate the means E[Xi] and E[Xj] using their marginal distributions.
        4. Calculate the expected value of the product E[Xi*Xj] using the joint marginal distribution.
        5. Calculate the covariance using the formula Cov(Xi, Xj) = E[Xi*Xj] - E[Xi]*E[Xj].

        IMPORTANT: Normalisation must be ensured throughout the calculations. We asusme i_marginal and j_marginal are normalised.
    """

    # Calcuate appropriat joint marginal distribution
    joint_marginal = np.sum(data, axis = tuple(k for k in [0, 1, 2, 3] if k not in [i, j]))

    # Normalise joint marginal
    joint_marginal /= np.sum(joint_marginal)

    # Extract the needed marginal means
    index_to_key = {0: 'x', 1: 'y', 2: 'z', 3: 'w'}
    i_mean = marginals[index_to_key[i]]["mean"]
    j_mean = marginals[index_to_key[j]]["mean"]

    # Calculate the expected value E[Xi*Xj] using the joint marginal distribution
    E_Xi_Xj = np.sum(joint_marginal * np.outer(x - i_mean, x - j_mean))

    # Calculate the covariance using the formula Cov(Xi, Xj) = E[Xi*Xj] - E[Xi] * E[Xj]
    return E_Xi_Xj - (i_mean * j_mean)

if __name__ == "__main__":
    import sympy as sp

    a_sym, b_sym, c_sym = sp.symbols('a b c')

    cov_matrix_sym = sp.Matrix([
        [a_sym, 0, c_sym, 0],
        [0, a_sym, 0, -c_sym],
        [c_sym, 0, b_sym, 0],
        [0, -c_sym, 0, b_sym]]
    )

    cov_matrix = np.array(cov_matrix_sym.subs({a_sym: 1, b_sym: 1.2, c_sym: 0.4})).astype(np.float64)

    data = generate_data_4D(cov_matrix)

    marginals = calculate_4D_marginals(data)
    
    # Generate recovered covariance matrix
    index_to_key = {0: 'x', 1: 'y', 2: 'z', 3: 'w'}
    recovered_cov_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if j == i:
                recovered_cov_matrix[i, j] = marginals[index_to_key[i]]["variance"]
                continue

            recovered_cov_matrix[i, j] = calculate_4D_covariance_element(data, marginals, i, j)
    
    print(np.round(recovered_cov_matrix, 3))