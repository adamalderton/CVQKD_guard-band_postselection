import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from numba import njit
from scipy.stats import multivariate_normal, norm
import time
from pprint import pprint
import numpy as np
import sympy as sp
import seaborn as sb

class GBSR_quantum_statistics():
    def __init__(
            self,
            modulation_variance,
            transmittance,
            excess_noise,
            grid_range = [-10, 10],
            num_points_on_axis = 64,
            JIT = True,
            evaluate_4D = False
        ) -> None:
        """
        Class arguments:
            modulation_variance: Alice's modulation variance $V_\\text{mod}$.
            transmittance: Transmissivity $T$ of the channel.
            excess_noise: Excess noise $\\xi$ of the channel.
        """

        self.modulation_variance = modulation_variance
        self.transmittance = transmittance
        self.excess_noise = excess_noise

        self.JIT = JIT
        self.evaluate_4D = evaluate_4D

        self.alice_variance = modulation_variance + 1.0                              # Alice's effective variance $V_A = $V_\\text{mod} + 1$ in SNU.
        self.bob_variance = (transmittance * modulation_variance) + 1 + excess_noise # Bob's effective variance $V_B = T V_\\text{mod} + 1 + \\xi$ in SNU.

        # Covariance matrix coefficients derived from the above
        self.a = self.alice_variance
        self.b = self.bob_variance
        self.c = np.sqrt(self.transmittance * (self.alice_variance*self.alice_variance - 1))

        # Initialise covariance matrix and its coefficients a, b and c using sympy. Numerical values can be substituted in later.
        self.a_sym, self.b_sym, self.c_sym = sp.symbols('a b c', real = True, positive = True)

        # TODO: Change this to uneven grid spacing.
        self.num_points_on_axis = num_points_on_axis
        self.axis_values = np.linspace(grid_range[0], grid_range[1], self.num_points_on_axis)

        if evaluate_4D:
            # As we are evaluating the full 4D versions of the functions, we assign the 4D a_PS etc method
            self.evaluate_a_PS_b_PS_c_PS = self._evaluate_a_PS_b_PS_c_PS_4D

            self.cov_mat_sym = sp.Matrix([
                [self.a_sym, 0, self.c_sym, 0],
                [0, self.a_sym, 0, -self.c_sym],
                [self.c_sym, 0, self.b_sym, 0],
                [0, -self.c_sym, 0, self.b_sym]
            ])

            # Define symbols for Alice and Bob's coherent states.
            self.alpha_sym, self.beta_sym = sp.symbols('alpha beta', complex = True)
            
            # Define symbols for the real and complex parts of Alice and Bob's coherent states.
            self.alpha_re_sym, self.alpha_im_sym = sp.symbols('alpha_re alpha_im', real = True)
            self.beta_re_sym, self.beta_im_sym = sp.symbols('beta_re beta_im', real = True)

            # Generate the grid over which to perform necessary numerical integrations.
            # TODO: Change this to uneven grid spacing.
            self.alpha_re_mesh, self.alpha_im_mesh, self.beta_re_mesh, self.beta_im_mesh = np.meshgrid(self.axis_values, self.axis_values, self.axis_values, self.axis_values, indexing = "ij")

            # Generate Q_star_values. These are constant for now and do not need to be updated (for constant a, b and c)
            self.Q_4_star_lambda = multivariate_normal(
                mean = np.zeros(4),
                cov = self.cov_mat_sym.subs({self.a_sym: self.a, self.b_sym: self.b, self.c_sym: self.c}) + np.eye(4)
            ).pdf
            self.Q_4_star_values = self.Q_4_star_lambda(np.stack((self.alpha_re_mesh, self.alpha_im_mesh, self.beta_re_mesh, self.beta_im_mesh), axis = -1))
            self.Q_4_star_values /= np.sum(self.Q_4_star_values)
        
        else: # If we are not evaluating the 4D functions, we can simply use the 2D variations
            self.evaluate_a_PS_b_PS_c_PS = self._evaluate_a_PS_b_PS_c_PS_2D

            self.cov_mat_sym = sp.Matrix([
                [self.a_sym, self.c_sym],
                [self.c_sym, self.b_sym]
            ])

            # Define the appropriate meshgrid for the 2D case. That is, x_mesh and y_mesh
            # TODO: Change this to uneven grid spacing.
            self.x_mesh, self.y_mesh = np.meshgrid(self.axis_values, self.axis_values, indexing = "ij")

            self.Q_2_star_lambda = multivariate_normal(
                mean = np.zeros(2),
                cov = self.cov_mat_sym.subs({self.a_sym: self.a, self.b_sym: self.b, self.c_sym: self.c}) + np.eye(2)
            ).pdf
            self.Q_2_star_values = self.Q_2_star_lambda(np.stack((self.x_mesh, self.y_mesh), axis = -1))
            self.Q_2_star_values /= np.sum(self.Q_2_star_values)

        # Placeholder attributes for those that need to be evaluated with the specifics of the guard bands in mind.
        self.p_pass = None              # Numerical value of the probability of passing the filter function. 
        self.px_PS = None               # Array containing post-selection marginal probability distribution values p(X = x).
        self.py_PS = None               # Array containing post-selection marginal probability distribution values p(Y = y).

        self.a_PS = None                # Effective covariance matrix coefficient a_PS.
        self.b_PS = None                # Effective covariance matrix coefficient b_PS.
        self.c_PS = None                # Effective covariance matrix coefficient c_PS.

        self.Q_2_PS_values = None       # Array containing post-selection joint probability distribution values p(X = x, Y = y). Equivalent to pxy_PS.
        self.cov_mat_2_PS = None        # Effective covariance matrix of the (2D marginal of the) post-selected state.

        self.Q_4_PS_values = None       # Array of values of the Q function AFTER post-selection on a grid of points.
        self.cov_mat_4_PS = None        # Effective covariance matrix of the post-selected state.
        self.marginals_4_PS = []        # List of marginal probability distributions for alpha_re, alpha_im, beta_re and beta_im.
        self.marginal_4_PS_means = []   # List of means for the marginal probability distributions for alpha_re, alpha_im, beta_re and beta_im.

    def evaluate_a_PS_b_PS_c_PS(self, tau_arr, g_arr):
        raise NotImplementedError("This method should be overridden by the 2D or 4D versions of the method.")

    def plot_marginals_2_PS(self):
        """
        Plot the marginal distributions and heatmap of joint distribution for postselected data.
        This method plots the marginal distributions p(x) and p(y), as well as the heatmap of the joint distribution p(x, y)
        for postselected data.
        Returns:
            None
        """
        
        if self.px_PS is None:
            raise ValueError("Marginals and joint distribution have not been evaluated yet. Please evaluate them first.")
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot px
        axs[0].plot(self.axis_values, self.px_PS)
        axs[0].set_title('Plot of px')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('p(x = X)')

        # Plot py
        axs[1].plot(self.axis_values, self.py_PS)
        axs[1].set_title('Plot of py')
        axs[1].set_xlabel('y')
        axs[1].set_ylabel('p(y = Y)')

        # Plot pxy using imshow, with the range set as axis_range \times axis_range
        axs[2].imshow(self.Q_2_PS_values, extent = (self.axis_values[0], self.axis_values[-1], self.axis_values[0], self.axis_values[-1]), origin='lower')
        axs[2].set_title('Heatmap of p(x = X, y = Y)')

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plot
        plt.show()

    def _evaluate_p_pass_2_and_Q_2_PS_values(self, tau_arr, g_arr):
        """
            Evaluate the probability of passing the filter function by integrating over the 2D marginalised version: (F(y) \\times Q*(x, y)).

            Arguments:
                tau_arr: array(float)
                    An array holding the values of $\\tau_i$.
                g_arr: array(float)
                    An array holding the values of $g_{\\pm, i}$. g[i][0] contains $g_{i,-}$ and g[i][1] contains $g_{i,+}$.
        """
        # Generate the filter function $F(y)$ with the given guard band properties.
        F_lambda = self._generate_F_2_lambda(tau_arr, g_arr)

        # Apply the filter function. Q_PS(x, y) = F(y) * Q*(x, y)
        self.Q_2_PS_values = F_lambda(self.y_mesh) * self.Q_2_star_values

        # Find p_pass by integrating the 'remaining' unnormalised pdf values.
        self.p_pass = self._integral_over_2D_pdf_values(self.Q_2_PS_values)

        # Divide through by p_pass to find normalise Q_PS_values
        self.Q_2_PS_values /= self.p_pass

        return self.p_pass, self.Q_2_PS_values

    def _generate_F_2_lambda(self, tau_arr, g_arr):
        """
            Generate a function which takes a real number y as an argument that acts as the filter function F.

            Arguments:
                tau_arr: np.array(float)
                    A numpy array holding the values of $\\tau_i$.
                g_arr: np.array(float)
                    A numpy array holding the values of $g_{\\pm, i}$. g[i][0] contains $g_{i,-}$ and g[i][1] contains $g_{i,+}$.
        """
        def filter_function(y):
            y_minus_tau = y - tau_arr

            # Check if y is within any guard band bounds
            condition = (-g_arr[:, 0] <= y_minus_tau) & (y_minus_tau <= g_arr[:, 1])

            # If y is in any guard band, return 0
            if np.any(condition):
                return 0
            
            return 1
        
        # First, JIT compile the filter function if necessary
        if self.JIT:
            filter_function = njit(filter_function)
        
        # Then, vectorize the function using np.vectorize, and return
        return np.vectorize(filter_function)

    def _evaluate_Q_2_PS_marginals(self):
        """
            Given Q_2_PS_values, evaluate the marginal probability distributions of Q_PS p(X = x), p(Y = y). Q_2_PS = p(x = X, y = Y) of course.
        """
        if self.Q_2_PS_values is None:
            raise ValueError("Q_2_PS_values have not been evaluated yet. Please evaluate Q_2_PS_values first.")
        
        # Marginalise over the y axis to find p(x = X), with renormalisation
        self.px_PS = np.sum(self.Q_2_PS_values, axis = 1)
        self.px_PS /= np.sum(self.px_PS)

        # Marginalise over the x axis to find p(y = Y), with renormalisation
        self.py_PS = np.sum(self.Q_2_PS_values, axis = 0)
        self.py_PS /= np.sum(self.py_PS)

        return self.px_PS, self.py_PS

    def _evaluate_effective_covariance_matrix_2D(self):
        """
            Evaluate the effective covariance matrix of the post-selected state via finding (co)variances of the appropriate marginals.
        """
        
        # Need to evaluate marginals. Do this now, even if done before to ensure up-to-date values. It's not particularly expensive.
        self._evaluate_Q_2_PS_marginals()

        effective_husimi_cov_mat = np.zeros((2, 2))

        # Find variance of Q(x) marginal
        mean_px = np.sum(self.px_PS * self.axis_values)
        effective_husimi_cov_mat[0][0] = np.sum((self.axis_values - mean_px)**2 * self.px_PS)

        # Find variance of Q(y) marginal
        mean_py = np.sum(self.py_PS * self.axis_values)
        effective_husimi_cov_mat[1][1] = np.sum((self.axis_values - mean_py)**2 * self.py_PS)

        # Find covariance of Q(x, y) marginal
        effective_husimi_cov_mat[0][1] = np.sum((self.x_mesh - mean_px) * (self.y_mesh - mean_py) * self.Q_2_PS_values)
        effective_husimi_cov_mat[1][0] = effective_husimi_cov_mat[0][1]

        self.cov_mat_2_PS = effective_husimi_cov_mat - np.eye(2)

        return self.cov_mat_2_PS

    def _evaluate_a_PS_b_PS_c_PS_2D(self, tau_arr, g_arr):
        """
            Evaluate the effective covariance matrix coefficients for the given post-selected
            state represented in Q_2_PS_values. This involves finding the 2D covariance matrix
            which is of the form [[a, c], [c, b]].
        """
        # For the provided tau_arr and g_arr, calculate the appropriate Q_PS values:
        self._evaluate_p_pass_2_and_Q_2_PS_values(tau_arr, g_arr)

        # With Q_PS and p_pass evaluated, we can now find the effective covariance matrix.
        self.cov_mat_2_PS = self._evaluate_effective_covariance_matrix_2D()

        self.a_PS = self.cov_mat_2_PS[0][0]
        self.b_PS = self.cov_mat_2_PS[1][1]
        self.c_PS = self.cov_mat_2_PS[0][1]

        return self.a_PS, self.b_PS, self.c_PS

    def _integral_over_2D_pdf_values(self, integrand_values):
        return np.sum(integrand_values)

    def plot_marginals_4_PS(self):
        """
        Plots the marginal distributions and joint distribution for postselection.
        Raises:
            ValueError: If the marginal distributions and joint distribution have not been evaluated yet.
        """


        if self.px_PS is None or self.py_PS is None or self.pxy_PS is None:
            raise ValueError("Marginals and joint distribution have not been evaluated yet. Please evaluate them first.")

        # Set the figure size
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot px
        axs[0].plot(self.axis_values, self.px_PS)
        axs[0].set_title('Plot of px')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('px')

        # Plot py
        axs[1].plot(self.axis_values, self.py_PS)
        axs[1].set_title('Plot of py')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('py')

        # Plot pxy using imshow, with the range set as axis_range \times axis_range
        axs[2].imshow(self.pxy_PS, extent = (self.axis_values[0], self.axis_values[-1], self.axis_values[0], self.axis_values[-1]), origin='lower')
        axs[2].set_title('Heatmap of pxy')
        axs[2].set_xlabel('Index')
        axs[2].set_ylabel('Index')

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plot
        plt.show()

    def _evaluate_p_pass_4_and_Q_4_PS_values(self, tau_arr, g_arr):
        """
        Evaluate the probability of passing the filter function by integrating over (F(\\beta) \\times Q*(\\alpha, \\beta)) over the entire complex plane.

        Arguments:
            tau_arr: array(float)
                An array holding the values of $\\tau_i$.
            g_arr: array(float)
                An array holding the values of $g_{\\pm, i}$. g[i][0] contains $g_{i,-}$ and g[i][1] contains $g_{i,+}$.
        """
        # Generate the filter function $F(\beta)$ with the given guard band properties.
        F_lambda = self._generate_F_4_lambda(tau_arr, g_arr)

        # Find F * Q_star, in numerical form.
        self.Q_4_PS_values = F_lambda(self.beta_re_mesh, self.beta_im_mesh) * self.Q_4_star_values

        # Integrate over the entire complex plane (using a 4D trapz) to find p_pass
        self.p_pass = self._integral_over_4D_pdf_values(self.Q_4_PS_values)

        # Divide through by p_pass to find normalise Q_PS_values
        self.Q_4_PS_values /= self.p_pass

        return self.p_pass, self.Q_4_PS_values

    def _generate_F_4_lambda(self, tau_arr, g_arr):
        """
            Generate a function which takes one complex number as parameters that acts as the filter function F.

            Arguments:
                tau_arr: np.array(float)
                    A numpy array holding the values of $\\tau_i$.
                g_arr: np.array(float)
                    A numpy array holding the values of $g_{\\pm, i}$. g[i][0] contains $g_{i,-}$ and g[i][1] contains $g_{i,+}$.
        """

        def filter_function(b_real, b_imag):
            x_real = b_real - tau_arr
            x_imag = b_imag - tau_arr

            # Check if there exists any element within the guard band bounds for both real and imaginary parts
            condition_real = (-g_arr[:, 0] <= x_real) & (x_real <= g_arr[:, 1])
            condition_imag = (-g_arr[:, 0] <= x_imag) & (x_imag <= g_arr[:, 1])

            # If any condition is met in either real or imaginary part, return 0
            if np.any(condition_real) or np.any(condition_imag):
                return 0
            return 1
        
        # First, JIT compile the filter function
        if self.JIT:
            filter_function = njit(filter_function)
        
        # Then, vectorize the function using np.vectorize, and return
        return np.vectorize(filter_function)

    def _evaluate_Q_4_PS_marginals(self):
        """
            Evaluate the marginal probability distributions of Q_PS p(X = x), p(Y = y) and p(X = x, Y = y) using the joint probability distribution p(alpha_re, alpha_im, beta_re, beta_im).
        """
        if self.Q_4_PS_values is None:
            raise ValueError("Q_PS_4_values have not been evaluated yet. Please evaluate Q_PS_4_values first.")

        alpha_re_marginal = np.sum(self.Q_4_PS_values, axis = (1, 2, 3))
        alpha_im_marginal = np.sum(self.Q_4_PS_values, axis = (0, 2, 3))
        beta_re_marginal = np.sum(self.Q_4_PS_values, axis = (0, 1, 3))
        beta_im_marginal = np.sum(self.Q_4_PS_values, axis = (0, 1, 2))

        # Normalise
        alpha_re_marginal /= np.sum(alpha_re_marginal)
        alpha_im_marginal /= np.sum(alpha_im_marginal)
        beta_re_marginal /= np.sum(beta_re_marginal)
        beta_im_marginal /= np.sum(beta_im_marginal)

        # Assign marginals to px and py. For now, we choose the real elements.
        self.px_PS = alpha_re_marginal
        self.py_PS = beta_re_marginal

        # Calculate a pxy marginal via techniques used in self._evaluate_effective_covariance_matrix(). That is, integrate over the imaginary parts.
        self.pxy_PS = np.sum(self.Q_4_PS_values, axis = (1, 3))
        self.pxy_PS /= np.sum(self.pxy_PS)

        # Store resulting values
        self.marginals_4_PS = [alpha_re_marginal, alpha_im_marginal, beta_re_marginal, beta_im_marginal]
        self.marginal_4_PS_means = [np.sum(self.marginals_4_PS[i] * self.axis_values) for i in range(4)]

        return self.marginals_4_PS

    def _evaluate_effective_covariance_matrix_4D(self):
        """
            Evaluate the effective covariance matrix of the post-selected state via finding (co)variances of the appropriate marginals.
        """
        effective_husimi_cov_mat = np.zeros((4, 4))

        # Initialise and calculate marginals (and their means).
        self._evaluate_Q_4_PS_marginals()
        joint_marginal = np.zeros((self.num_points_on_axis, self.num_points_on_axis))
        
        for i in range(4):
            for j in range(4):
                if j == i:
                    # Leading diagonal elements are just variances of marginals
                    effective_husimi_cov_mat[i][i] = np.sum((self.axis_values - self.marginal_4_PS_means[i])**2 * self.marginals_4_PS[i])
                    continue

                # Off-diagonal elements are covariances of marginals. First, calculate joint marginal and normalise.
                joint_marginal = np.sum(self.Q_4_PS_values, axis = tuple(k for k in [0, 1, 2, 3] if k not in [i, j]))
                joint_marginal /= np.sum(joint_marginal)

                # Use joint marginal to calculate covariance
                effective_husimi_cov_mat[i][j] = np.sum(joint_marginal * np.outer(self.axis_values - self.marginal_4_PS_means[i], self.axis_values - self.marginal_4_PS_means[j])) - (self.marginal_4_PS_means[i] * self.marginal_4_PS_means[j])
        
        # The covariance matrix of the state is therefore the effective husimi covariance matrix minus the identity matrix.
        self.cov_mat_4_PS = effective_husimi_cov_mat - np.eye(4)

        return self.cov_mat_4_PS

    def _evaluate_a_PS_b_PS_c_PS_4D(self, tau_arr, g_arr):
        """
            Evaluate the effective covariance matrix coefficients for the given post-selected
            state represented in Q_4_PS_values. This involves finding the entire 4D covariance matrix.
        """
        # For the provided tau_arr and g_arr, calculate the appropriate Q_PS values:
        self._evaluate_p_pass_4_and_Q_4_PS_values(tau_arr, g_arr)

        # With Q_PS and p_pass evaluated, we can now find the effective covariance matrix.
        self.cov_mat_4_PS = self._evaluate_effective_covariance_matrix_4D()

        self.a_PS = self.cov_mat_4_PS[0][0]
        self.b_PS = self.cov_mat_4_PS[2][2]
        self.c_PS = self.cov_mat_4_PS[0][2]

        return self.a_PS, self.b_PS, self.c_PS

    def _integral_over_4D_pdf_values(self, integrand_values):
        return np.sum(integrand_values)
        
class GBSR(GBSR_quantum_statistics):
    def __init__(
            self,
            m,
            modulation_variance,
            transmittance,
            excess_noise,
            grid_range = [-10, 10],
            num_points_on_axis = 64,
            JIT = True,             # Just-In-Time compilation of computatinally expensive functions
            evaluate_4D = False     # A flag to signal the evaluation the 4D functions, Q_4_star etc. This is something like a debug flag.
        ) -> None:
        super().__init__(modulation_variance, transmittance, excess_noise, grid_range = grid_range, num_points_on_axis = num_points_on_axis, JIT = JIT, evaluate_4D = evaluate_4D)

        self.large_negative = -1e10
        self.large_positive = 1e10

        self.m = m
        self.number_of_intervals = 2**m

    def evaluate_key_rate_in_bits_per_pulse(
            self,
            tau_arr,
            g_arr,
            quantisation_entropy = None,
            classical_leaked_information = None,
            error_rate = None,
            holevo_information = None
        ):
        # print("evaluate_key_rate_in_bits_per_pulse")
        """
            Evaluate the key rate for GBSR with a given number of slices $m$ and an array holding each interval edge, and a 2D array holding the relative guard band spans.

            Can pass parts of the key rate calculation to avoid redundant calculations.
        """

        if quantisation_entropy is None:
            quantisation_entropy = self.evaluate_quantisation_entropy(tau_arr)
        
        if classical_leaked_information is None:
            if error_rate is None:
                error_rate = self.evaluate_error_rate(tau_arr, g_arr)
            
            classical_leaked_information = self._binary_entropy(error_rate)
        
        if holevo_information is None:
            self.evaluate_a_PS_b_PS_c_PS(tau_arr, g_arr)

            holevo_information = self._evaluate_holevo_information(self.a_PS, self.b_PS, self.c_PS)

        # holevo_information = self._evaluate_holevo_information(self.a, self.b, self.c)
        return self.p_pass * (quantisation_entropy - classical_leaked_information - holevo_information)
        #return quantisation_entropy - classical_leaked_information - holevo_information

    def evaluate_error_rate(self, tau_arr, g_arr):
        """
            Evaluate the error rate of the protocol.

            e = \\sum_{i = 0}^{2^m - 1} \\sum_{j = 0, j \neq i}^{2^m - 1} \\: \\int_{\tau_i}^{\tau_{i+1}} \\! dX \\int_{\tau_j + g_{j,+}}^{\tau_{j+1} - g_{j+1,-}} \\! dY \\: p_\text{joint}(X, Y) \\
        """

        error_rate = 0.0

        # The covariance matrix associated with the marginal joint probability distribution p(X, Y).
        xy_cov_mat = [
            [self.a, self.c],
            [self.c, self.b]
        ]
        
        for i in range(self.number_of_intervals):
            for j in range(self.number_of_intervals):
                if i == j:
                    continue

                # Define the limits of integration for X and Y
                xlims = [tau_arr[i], tau_arr[i + 1]]
                ylims = [tau_arr[j] + g_arr[j][1], tau_arr[j + 1] - g_arr[j + 1][0]]

                # Integrate the joint PDF over the given limits.
                error_rate += self._integrate_2D_gaussian_pdf(xlims, ylims, xy_cov_mat)

        return error_rate

    def evaluate_quantisation_entropy(self, tau_arr):
        """
            Evaluate the quantisation entropy of the protocol.

            H = - \\sum_{i = 0}^{2^m - 1} \\int_{\tau_i}^{\tau_{i+1}} \\! dX \\: p(X = x) \\log_2 p(X = x)
        """
        interval_probabilities = [self._integrate_1D_gaussian_pdf([tau_arr[i], tau_arr[i+1]], np.sqrt(self.alice_variance)) for i in range(self.number_of_intervals)]
        
        return -1.0 * np.sum([interval_probabilities[i] * np.log2(interval_probabilities[i]) for i in range(self.number_of_intervals)])

    def _evaluate_mutual_information(self):
        """
            Evaluate the mutual information between Alice and Bob's correlated Gaussian variables.
            This will later be done via numerical integration but the known analytical form can be read off for now.
        """
        pass

    def _evaluate_holevo_information(self, a, b, c):
        """
            An upper bound for the Holevo information using the calculated covariance matrix. 
            See first-year report or (Laudenbach 2018) for more details.
            We can directly find the symplectic eigenvalues of the needed covariance matrix by using the familiar formulae.
        """
        nu = np.zeros(3)

        sqrt_value = (a + b)**2 - (4.0 * c**2)

        # if sqrt_value < 0:
        #     # TODO: This is a hack for now just to keep things going.
        #     sqrt_value = 0.0

        nu[0] = 0.5 * (np.sqrt(sqrt_value) + (b - a))
        nu[1] = 0.5 * (np.sqrt(sqrt_value) - (b - a))

        # Find the symplectic eigenvalue of the covariance matrix describing A conditioned on B.
        nu[2] = a - ((c**2) / (b + 1))

        # With all the necessary symplectic eigenvalues, we can now find the Holevo information:
        return self._g(nu[0]) + self._g(nu[1]) - self._g(nu[2])

    def _g(self, x):
        """
            Needed to find the correct contrubution to the Holevo information for each symplectic eigenvalue from the appropriate covariance matrix.
        """

        # x may be less than one due to integration truncation etc. This is usually unintended, so we can evaluate g(x = 1) which is 0.0.
        if x <= 1.0:
            return 0.0

        return ((x + 1.0) / 2.0) * np.log2((x + 1.0) / 2.0) - ((x - 1.0) / 2.0) * np.log2((x - 1.0) / 2.0)

    def _integrate_1D_gaussian_pdf(self, lims, std_dev, mean = 0.0) -> float:
        """
            Find the integral of a 1D Gaussian probability density function (PDF) within the given limits.
            Parameters:
                lims (list): A list containing the lower and upper limits of integration.
                std_dev (float): The standard deviation of the Gaussian distribution.
                mean (float, optional): The mean of the Gaussian distribution. Default is 0.0.
            Returns:
                float: The integral of the Gaussian PDF within the specified limits.
        """

        # Replace any infinite limits with large numbers
        lims = [self.large_negative if x == -np.inf else x for x in lims]
        lims = [self.large_positive if x == np.inf else x for x in lims]

        rv = norm(loc = mean, scale = std_dev)
        return rv.cdf(lims[1]) - rv.cdf(lims[0])

    def _integrate_2D_gaussian_pdf(self, xlims, ylims, cov_matrix, mean=[0.0, 0.0]) -> float:
        """
        Calculates the integral of a 2D Gaussian probability density function (PDF) within the specified limits.
        Parameters:
        - xlims (list): The limits of the x-axis in the form [x1, x2].
        - ylims (list): The limits of the y-axis in the form [y1, y2].
        - cov_matrix (ndarray): The covariance matrix of the Gaussian distribution.
        - mean (list, optional): The mean of the Gaussian distribution. Default is [0.0, 0.0].
        Returns:
        - float: The integral of the 2D Gaussian PDF within the specified limits.
        """
        
        rv = multivariate_normal(mean = mean, cov = cov_matrix)

        # Replace any infinite limits with large numbers in xlims and ylims.
        xlims = [self.large_negative if x == -np.inf else x for x in xlims]
        xlims = [self.large_positive if x == np.inf else x for x in xlims]

        ylims = [self.large_negative if y == -np.inf else y for y in ylims]
        ylims = [self.large_positive if y == np.inf else y for y in ylims]

        # Extract the limits
        x1, x2 = xlims
        y1, y2 = ylims

        # Probability inside the rectangle defined by the limits
        return (rv.cdf([x2, y2]) - rv.cdf([x2, y1]) - rv.cdf([x1, y2]) + rv.cdf([x1, y1]))

    def _binary_entropy(self, e):
        one_minus_e = 1.0 - e
        return - (e * np.log2(e)) - (one_minus_e * np.log2(one_minus_e))

if __name__ == "__main__":
    from tqdm import tqdm

    # gbsr = GBSR(1, 2.2, 0.32, 0.02)
    gbsr = GBSR(1, 2.2, 0.6, 0.0, num_points_on_axis = 128)

    tau_arr = np.array([-np.inf, 0.0, np.inf])
    g_arr = np.array([[0.0, 0.0], [1.5, 1.5], [0.0, 0.0]])

    print(gbsr.evaluate_key_rate_in_bits_per_pulse(tau_arr, g_arr))

    