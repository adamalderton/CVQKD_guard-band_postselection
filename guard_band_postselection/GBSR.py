import numpy as np
import sympy as sp
# from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
from scipy.integrate import dblquad

class GBSR_quantum_statistics():
    """
        A class to evaluate the quantum-statistics elements of the GBSR protocol.
        Should be integrated into the GBSR class as a parent.
    """

    def __init__(
            self,
            modulation_variance,
            transmittance,
            excess_noise,
            progress_bar = False
    ):
        """
            Initialise the class with the given parameters.

            Class arguments:
                modulation_variance: Alice's modulation variance $V_\\text{mod}$.
                transmittance: Transmissivity $T$ of the channel.
                excess_noise: Excess noise $\\xi$ of the channel.
                JIT: A flag to signal the use of Just-In-Time compilation.
        """
        
        self.progress_bar = progress_bar

        self.modulation_variance = modulation_variance
        self.transmittance = transmittance
        self.excess_noise = excess_noise

        self.alice_variance = modulation_variance + 1.0                              # Alice's effective variance $V_A = $V_\\text{mod} + 1$ in SNU.
        self.bob_variance = (transmittance * modulation_variance) + 1 + excess_noise # Bob's effective variance $V_B = T V_\\text{mod} + 1 + \\xi$ in SNU.

        # Covariance matrix coefficients derived from the above
        self.a = self.alice_variance
        self.b = self.bob_variance
        self.c = np.sqrt(self.transmittance * (self.alice_variance*self.alice_variance - 1))

        # Initialise (2D marginalised) covariance matrix and its coefficients a, b and c using sympy. Numerical values can be substituted in later.
        self.a_sym, self.b_sym, self.c_sym = sp.symbols('a b c', real = True, positive = True)
        self.cov_mat_sym = sp.Matrix([
            [self.a_sym, self.c_sym],
            [self.c_sym, self.b_sym]
        ])

        # Initialise 'random variables' to integrate over.
        self.px_rv = norm(loc = 0.0, scale = np.sqrt(self.alice_variance)) # Alice's random variable
        self.py_rv = norm(loc = 0.0, scale = np.sqrt(self.bob_variance))   # Bob's random variable
        self.Q_star_rv = multivariate_normal(mean = np.zeros(2), cov = np.array([[self.a, self.c], [self.c, self.b]])) # The joint random variable Q*.

        # Placeholder attributes for those that need to be evaluated with the specifics of the guard bands in mind.
        self.p_pass = None              # Numerical value of the probability of passing the filter function. 
        self.a_PS = None                # Effective covariance matrix coefficient a_PS.
        self.b_PS = None                # Effective covariance matrix coefficient b_PS.
        self.c_PS = None                # Effective covariance matrix coefficient c_PS.
        self.cov_mat_PS = None        # Effective covariance matrix of the (2D marginal of the) post-selected state.

        # Placeholder attributes for arrays that hold marginal distribution values, FOR PLOTTING PURPOSES ONLY. These should not be used as part of any numerics.
        self.px_PS_values = None        # Array containing post-selection marginal probability distribution values p(X = x).
        self.py_PS_values = None        # Array containing post-selection marginal probability distribution values p(Y = y).
        self.Q_PS_values = None         # Array containing post-selected joint probability distribution values p(X = x, Y = y).

    def plot_marginals(self, tau_arr, g_arr, axis_range = [-10, 10], num_points_on_axis = 100):
        """
            Plot the marginal distributions and heatmap of joint distribution for postselected data.
            This method plots the marginal distributions p(x) and p(y), as well as the heatmap of the joint distribution p(x, y)
            for postselected data.

            Returns:
                None
        """
        if self.px_PS_values is None:
            self._evaluate_marginals(tau_arr, g_arr, axis_range, num_points_on_axis)
        
        axis_values = np.linspace(axis_range[0], axis_range[1], num_points_on_axis)
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot px
        axs[0].plot(axis_values, self.px_PS_values)
        axs[0].set_title('Plot of px')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('p(x = X)')

        # Plot py
        axs[1].plot(axis_values, self.py_PS_values)
        axs[1].set_title('Plot of py')
        axs[1].set_xlabel('y')
        axs[1].set_ylabel('p(y = Y)')

        # Plot pxy using imshow, with the range set as axis_range \times axis_range
        axs[2].imshow(self.Q_PS_values, extent = (axis_values[0], axis_values[-1], axis_values[0], axis_values[-1]), origin = 'lower')
        axs[2].set_title('p(x = X, y = Y)')

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plot
        plt.show()

    def evaluate_p_pass(self, tau_arr, g_arr):
        """
            Evaluate the probability of passing the filter function by integrating over the 2D marginalised version: (F(y) \\times Q*(x, y)).

            Arguments:
                tau_arr: array(float)
                    An array holding the values of $\\tau_i$.
                g_arr: array(float)
                    An array holding the values of $g_{\\pm, i}$. g[i][0] contains $g_{i,-}$ and g[i][1] contains $g_{i,+}$.
        """
        self.p_pass = sum(self._integrate_1D_gaussian_pdf(self.py_rv, [tau_arr[i] + g_arr[i][1], tau_arr[i + 1] - g_arr[i][0]]) for i in range(len(tau_arr) - 1))
        return self.p_pass

    def evaluate_cov_mat_PS(self, tau_arr, g_arr):
        """
            Evaluate the covariance matrix of the post-selected state.
            This is done by evaluating the effective covariance matrix coefficients a_PS, b_PS and c_PS.
            This is the main method to be called to interact with this class.

            IMPORTANT: This method assums E(x) = E(y) = 0.0. This assumption is valid for the GBSR protcol due to symmetry. This assumption is made to minimise numerical integrations.

            Arguments:
                tau_arr: np.array(float)
                    A numpy array holding the values of $\\tau_i$.
                g_arr: np.array(float)
                    A numpy array holding the values of $g_{\\pm, i}$. g[i][0] contains $g_{i,-}$ and g[i][1] contains $g_{i,+}$.
        """
        e_x, var_x, e_y, var_y, cov_xy = self._compute_Q_PS_moments(self.Q_star_rv, tau_arr, g_arr)

        # Populate the covariance matrix of the Husimi-Q function for the post-selected state.
        effective_husimi_cov_mat = np.zeros((2, 2))
        effective_husimi_cov_mat[0][0] = var_x
        effective_husimi_cov_mat[1][1] = var_y
        effective_husimi_cov_mat[0][1] = cov_xy
        effective_husimi_cov_mat[1][0] = cov_xy

        # Calculate covariance matrix of the post-selected state.
        self.cov_mat_PS = effective_husimi_cov_mat - np.eye(2)

        # Extract a_PS, b_PS, and c_PS from the covariance matrix.
        self.a_PS = self.cov_mat_PS[0][0]
        self.b_PS = self.cov_mat_PS[1][1]
        self.c_PS = self.cov_mat_PS[0][1]

        return self.cov_mat_PS
    
    def _evaluate_marginals(self, tau_arr, g_arr, axis_range, num_points_on_axis):
        """
            Evaluate the marginal distributions for post-selected data.
            This is done by evaluating the marginal distributions p(x) and p(y) for post-selected data.
        """
        axis_values = np.linspace(axis_range[0], axis_range[1], num_points_on_axis)

        # Define the filter function
        def filter_function(y):
            # Start with a mask of ones (True)
            mask = np.ones_like(y, dtype=bool)
            for i in range(len(tau_arr)):
                y_minus_tau = y - tau_arr[i]
                condition = (-g_arr[i][0] <= y_minus_tau) & (y_minus_tau <= g_arr[i][1])
                # Update mask: set to False where condition is True
                mask &= ~condition
            return mask.astype(int)  # Convert boolean mask to int (1 or 0)

        # Find Q_PS_values first, then sum over rows and columns (and normalise) to find px_PS_values and py_PS_values.

        # Generate a meshgrid for the 2D joint distribution
        x_mesh, y_mesh = np.meshgrid(axis_values, axis_values)

        # Evaluate Q_star(x, y) over the meshgrid
        Q_star_values = self.Q_star_rv.pdf(np.dstack((x_mesh, y_mesh)))

        # Apply the filter function to Q_star_values, and normalise
        self.Q_PS_values = filter_function(y_mesh) * Q_star_values
        self.Q_PS_values /= np.sum(self.Q_PS_values)

        # Sum over rows and columns to find px_PS_values and py_PS_values, and renormalise
        self.px_PS_values = np.sum(self.Q_PS_values, axis=0)
        self.py_PS_values = np.sum(self.Q_PS_values, axis=1)
        self.px_PS_values /= np.sum(self.px_PS_values)
        self.py_PS_values /= np.sum(self.py_PS_values)

        return self.px_PS_values, self.py_PS_values, self.Q_PS_values

    def _compute_Q_PS_moments(self, rv, tau_arr, g_arr):
        """
            Compute the mean, variance, and covariance of x and y over specified limits for a 2D Gaussian distribution,
            considering exclusion zones (guard bands).

            Parameters:
                rv: scipy.stats.multivariate_normal object representing the 2D Gaussian distribution.
                tau_arr: Central values for guard bands.
                g_arr: Widths of guard bands around each tau.

            Returns:
                Means, variances, and covariance of x and y.
        """

        xlims = [-np.inf, np.inf]
        ylims = [-np.inf, np.inf]

        # Build guard band ranges in absolute terms
        band_ranges = []
        for i in range(len(tau_arr)):
            band_lower = tau_arr[i] - g_arr[i][0]
            band_upper = tau_arr[i] + g_arr[i][1]
            band_ranges.append((band_lower, band_upper))

        # Initialize the list of allowed intervals with the initial limits
        interval_list = [[ylims[0], ylims[1]]]

        # Exclude the guard bands from interval_list, like in _integrate_Q_PS
        for band_lower, band_upper in band_ranges:
            new_intervals = []
            for start, end in interval_list:
                # Exclude guard bands
                if band_upper <= start or band_lower >= end:
                    new_intervals.append([start, end])
                    continue
                if band_lower > start:
                    new_intervals.append([start, band_lower])
                if band_upper < end:
                    new_intervals.append([band_upper, end])
            interval_list = new_intervals

        # Initialize numerators and denominator
        E_X_num = 0.0
        E_X2_num = 0.0
        E_Y_num = 0.0
        E_Y2_num = 0.0
        E_XY_num = 0.0  # For covariance
        norm_const = 0.0

        if self.progress_bar:
            interval_list = tqdm(interval_list)

        # Integrate over the allowed intervals
        for y_start, y_end in interval_list:
            if y_end > y_start:
                # Update normalization constant, by simply integrating p(x, y) over the allowed intervals
                norm_const += self._integrate_2D_gaussian_pdf(rv, xlims, [y_start, y_end])

                # Define integrands
                def integrand_x(y, x):
                    return x * rv.pdf([x, y])

                def integrand_x2(y, x):
                    return x**2 * rv.pdf([x, y])

                def integrand_y(y, x):
                    return y * rv.pdf([x, y])

                def integrand_y2(y, x):
                    return y**2 * rv.pdf([x, y])

                def integrand_xy(y, x):
                    return x * y * rv.pdf([x, y])

                # Perform numerical integration
                E_X_num += dblquad(
                    integrand_x, xlims[0], xlims[1], lambda x: y_start, lambda x: y_end
                )[0]

                E_X2_num += dblquad(
                    integrand_x2, xlims[0], xlims[1], lambda x: y_start, lambda x: y_end
                )[0]

                E_Y_num += dblquad(
                    integrand_y, xlims[0], xlims[1], lambda x: y_start, lambda x: y_end
                )[0]

                E_Y2_num += dblquad(
                    integrand_y2, xlims[0], xlims[1], lambda x: y_start, lambda x: y_end
                )[0]

                E_XY_num += dblquad(
                    integrand_xy, xlims[0], xlims[1], lambda x: y_start, lambda x: y_end
                )[0]

        if norm_const == 0.0:
            raise ValueError("Normalization constant is zero. Check the integration limits and guard bands.")

        # Compute expected values and variances
        E_X = E_X_num / norm_const
        E_X2 = E_X2_num / norm_const
        Var_X = E_X2 - E_X**2

        E_Y = E_Y_num / norm_const
        E_Y2 = E_Y2_num / norm_const
        Var_Y = E_Y2 - E_Y**2

        # Compute covariance
        E_XY = E_XY_num / norm_const
        Cov_XY = E_XY - E_X * E_Y

        return E_X, Var_X, E_Y, Var_Y, Cov_XY

    def _integrate_Q_PS(self, xlims, ylims, tau_arr, g_arr):
        """
        Integrate the joint probability distribution of Q* and P_ps over the given limits,
        considering exclusion zones (guard bands) defined by tau_arr and g_arr.

        Arguments:
            xlims: list(float)
                The limits of integration for the x-axis.
            ylims: list(float)
                The limits of integration for the y-axis.
            tau_arr: list(float)
                Array of central values for the guard bands.
            g_arr: list(tuple(float, float))
                Array of tuples representing the lower and upper guard band widths around each tau.

        Returns:
            float: The result of the integration.
        """
        integral_result = 0.0

        y_lower = ylims[0]
        y_upper = ylims[1]

        # Build guard band ranges, in absolute terms (not relative to tau)
        band_ranges = []
        for i in range(len(tau_arr)):
            band_lower = tau_arr[i] - g_arr[i][0]
            band_upper = tau_arr[i] + g_arr[i][1]
            band_ranges.append((band_lower, band_upper))

        # Initialize the list of allowed intervals with the initial limits
        interval_list = [[y_lower, y_upper]]

        # Exclude the guard bands from interval_list
        for band_lower, band_upper in band_ranges:
            new_intervals = []

            for start, end in interval_list:

                # If the guard band does not overlap with [start, end], keep the interval as is
                if band_upper <= start or band_lower >= end:
                    new_intervals.append([start, end])
                    continue

                # The guard band overlaps with the interval, therefore split the interval into up to two intervals, excluding the guard band

                if band_lower > start:
                    new_intervals.append([start, band_lower])
                if band_upper < end:
                    new_intervals.append([band_upper, end])

                # If the guard band covers the whole interval, we don't add any interval

            # Update the interval list
            interval_list = new_intervals

        # Integrate over the allowed intervals
        for start, end in interval_list:
            if (end > start):
                integral_result += (1.0 / self.p_pass) * self._integrate_2D_gaussian_pdf(self.Q_star_rv, xlims, [start, end])

        return integral_result

    def _integrate_1D_gaussian_pdf(self, rv, lims) -> float:
        """
        Integrate the 1D Gaussian PDF over the specified limits.

        Parameters:
            rv: The random variable described by a 1D Gaussian distribution (scipy.stats.norm object).
            lims: A list or tuple with the limits of integration [lower_limit, upper_limit].
        """
        x1 = lims[0] if lims[0] != -np.inf else -1e10
        x2 = lims[1] if lims[1] != np.inf else 1e10

        return rv.cdf(lims[1]) - rv.cdf(lims[0])

    def _integrate_2D_gaussian_pdf(self, rv, xlims, ylims) -> float:
        """
        Integrate the 2D Gaussian PDF over the rectangle defined by xlims and ylims.

        Parameters:
            rv: The random variable described by a 2D Gaussian distribution (scipy.stats.multivariate_normal object).
            xlims: A list or tuple with the limits of integration for the x-axis [x_lower, x_upper].
            ylims: A list or tuple with the limits of integration for the y-axis [y_lower, y_upper].
        """
        # Extract limits and set to a large magnitude if infinite
        x1 = xlims[0] if xlims[0] != -np.inf else -1e10
        x2 = xlims[1] if xlims[1] != np.inf else 1e10
        y1 = ylims[0] if ylims[0] != -np.inf else -1e10
        y2 = ylims[1] if ylims[1] != np.inf else 1e10

        # Compute the probability inside the rectangle defined by the limits
        return (rv.cdf([x2, y2]) - rv.cdf([x2, y1]) - rv.cdf([x1, y2]) + rv.cdf([x1, y1]))

class GBSR(GBSR_quantum_statistics):
    def __init__(
            self,
            m,
            modulation_variance,
            transmittance,
            excess_noise,
            progress_bar = False
        ) -> None:
        super().__init__(modulation_variance, transmittance, excess_noise, progress_bar)

        self.m = m
        self.number_of_intervals = 2**m

        self.naive_holevo_information = self._evaluate_holevo_information(self.a, self.b, self.c)

    def evaluate_key_rate_in_bits_per_pulse(
            self,
            tau_arr,
            g_arr,
            quantisation_entropy = None,
            classical_leaked_information = None,
            error_rate = None,
            holevo_information = None,
            p_pass = None
        ):
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

        if p_pass is None:
            p_pass = self.evaluate_p_pass(tau_arr, g_arr)
        
        if holevo_information is None:
            # Also evaluates a_PS etc, which are elements of the covariance matrix of the post-selected state.
            self.evaluate_cov_mat_PS(tau_arr, g_arr)

            holevo_information = self._evaluate_holevo_information(self.a_PS, self.b_PS, self.c_PS)

        return self.p_pass * (quantisation_entropy - classical_leaked_information - holevo_information)

    def evaluate_error_rate(self, tau_arr, g_arr):
        """
            Evaluate the error rate of the protocol. Note that, of course, this operates on the statistics BEFORE post-selection.

            e = \\sum_{i = 0}^{2^m - 1} \\sum_{j = 0, j \neq i}^{2^m - 1} \\: \\int_{\tau_i}^{\tau_{i+1}} \\! dX \\int_{\tau_j + g_{j,+}}^{\tau_{j+1} - g_{j+1,-}} \\! dY \\: p_\text{joint}(X, Y) \\
        """

        error_rate = 0.0

        for i in range(self.number_of_intervals):
            for j in range(self.number_of_intervals):
                if i == j:
                    continue

                # Define the limits of integration for X and Y
                xlims = [tau_arr[i], tau_arr[i + 1]]
                ylims = [tau_arr[j] + g_arr[j][1], tau_arr[j + 1] - g_arr[j + 1][0]]

                # Integrate the joint PDF over the given limits.
                error_rate += self._integrate_2D_gaussian_pdf(self.Q_star_rv, xlims, ylims)

        return error_rate

    def evaluate_quantisation_entropy(self, tau_arr):
        """
            Evaluate the quantisation entropy of the protocol.

            H = - \\sum_{i = 0}^{2^m - 1} \\int_{\tau_i}^{\tau_{i+1}} \\! dX \\: p(X = x) \\log_2 p(X = x)
        """
        interval_probabilities = [self._integrate_1D_gaussian_pdf(self.px_rv, [tau_arr[i], tau_arr[i+1]]) for i in range(self.number_of_intervals)]
        
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

        sqrt_value = max(0, (a + b)**2 - (4.0 * c**2))

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

    def _binary_entropy(self, e):
        one_minus_e = 1.0 - e
        return - (e * np.log2(e)) - (one_minus_e * np.log2(one_minus_e))

if __name__ == "__main__":
    from tqdm import tqdm

    gbsr = GBSR(1, 2.2, 0.6, 0.0, progress_bar=True)

    tau_arr = [-np.inf, 0.0, np.inf]
    g_arr = [
        [0.0, 0.0],
        [0.5, 0.5],
        [0.0, 0.0]
    ]

    # gbsr.plot_marginals(tau_arr, g_arr, progress_bar=True)

    print(gbsr.evaluate_key_rate_in_bits_per_pulse(tau_arr, g_arr))

    