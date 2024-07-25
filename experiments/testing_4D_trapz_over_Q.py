import numpy as np
from scipy.stats import multivariate_normal

# Hardcoded covariance matrix with values in the range (0.0, 1.0)
Sigma = np.array([
    [0.8, 0.2, 0.1, 0.3],
    [0.2, 0.7, 0.2, 0.1],
    [0.1, 0.2, 0.9, 0.4],
    [0.3, 0.1, 0.4, 0.6]
])

# Define the mean vector (zero mean in all dimensions)
mean = np.zeros(4)

# Create a grid of points
grid_size = 50  # Define the range and density of the grid
x = np.linspace(-5, 5, grid_size)
y = np.linspace(-5, 5, grid_size)
z = np.linspace(-5, 5, grid_size)
w = np.linspace(-5, 5, grid_size)
X, Y, Z, W = np.meshgrid(x, y, z, w, indexing='ij')

# Flatten the grids and stack them to form a 2D array of shape (n_points, 4)
pos = np.empty(X.shape + (4,))
pos[:, :, :, :, 0] = X
pos[:, :, :, :, 1] = Y
pos[:, :, :, :, 2] = Z
pos[:, :, :, :, 3] = W
flat_pos = pos.reshape(-1, 4)

# Calculate the Gaussian PDF values on the grid
rv = multivariate_normal(mean, Sigma)
pdf_values = rv.pdf(flat_pos)

# Reshape the PDF values back to the grid shape
pdf_values = pdf_values.reshape(grid_size, grid_size, grid_size, grid_size)

# Function to integrate over the entire 4D space using the trapezoidal rule
def _4D_trapz_over_Q(integrand_values):
    integral = np.trapz(integrand_values, x=x)
    integral = np.trapz(integral, x=x)
    integral = np.trapz(integral, x=x)
    integral = np.trapz(integral, x=x)
    return integral

# Perform the integration
integral_result = _4D_trapz_over_Q(pdf_values)

print(f"Covariance matrix:\n{Sigma}")
print(f"Integral result: {integral_result}")

for i in [X, Y, Z, W]:
    for j in [X, Y, Z, W]:
        result = _4D_trapz_over_Q(i * j * pdf_values)
        print(f"{result:.2g} ", end="")
    print("")
