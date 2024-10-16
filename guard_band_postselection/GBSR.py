import numpy as np
import sympy as sp
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
from scipy.integrate import dblquad
from scipy.spatial.distance import hamming

class GBSR_quantum_statistics():
    """
        A class to evaluate the quantum-statistics elements of the GBSR protocol.
        Should be integrated into the GBSR class as a parent.
    """

    def __init__(self, modulation_variance, transmittance, excess_noise, progress_bar = False):
        """
            Initialise the class with the given parameters.

            Class arguments:
                modulation_variance: Alice's modulation variance $V_\\text{mod}$.
                transmittance: Transmissivity $T$ of the channel.
                excess_noise: Excess noise $\\xi$ of the channel.
        """
        
        self.progress_bar = progress_bar

        self.modulation_variance = modulation_variance
        self.transmittance = transmittance
        self.excess_noise = excess_noise

        self.alice_variance = modulation_variance + 1.0                                # Alice's effective variance $V_A = $V_\\text{mod} + 1$ in SNU.
        self.bob_variance = (transmittance * modulation_variance) + 1.0 + excess_noise # Bob's effective variance $V_B = T V_\\text{mod} + 1 + \\xi$ in SNU.

        self.SNR = (transmittance * modulation_variance) / (1.0 + excess_noise)        # Signal-to-noise ratio of the channel.
        self.I_AB = 0.5 * np.log2(1.0 + self.SNR)

        # Covariance matrix coefficients derived from the above
        self.a = self.alice_variance
        self.b = self.bob_variance
        self.c = np.sqrt(self.transmittance * (self.alice_variance**2 - 1))
        self.cov_mat = np.array([
            [self.a, self.c],
            [self.c, self.b]
        ])

        
        # Initialise 'random variables' for Alice and Bob respectively. These should be integrated over with methods .pdf(x) and .cdf(x) etc.
        self.px_rv = norm(loc = 0.0, scale = np.sqrt(self.alice_variance)) # Alice's random variable
        self.py_rv = norm(loc = 0.0, scale = np.sqrt(self.bob_variance))   # Bob's random variable
        self.joint_rv = multivariate_normal(mean = np.zeros(2), cov = self.cov_mat) # The joint random variable.
        
        # Substitute a, b and c in the covariance matrix, and add np.eye(2) as this is the Husimi-Q function covariance matrix, NOT the state covariance matrix.
        Q_star_cov_mat = self.cov_mat + np.eye(2)
        self.Q_star_rv = multivariate_normal(mean = np.zeros(2), cov = Q_star_cov_mat) # The joint random variable Q*.

        # Placeholder attributes for those that need to be evaluated with the specifics of the guard bands in mind.
        self.p_pass = None              # Numerical value of the probability of passing the filter function. 
        self.a_PS = None                # Effective covariance matrix coefficient a_PS.
        self.b_PS = None                # Effective covariance matrix coefficient b_PS.
        self.c_PS = None                # Effective covariance matrix coefficient c_PS.
        self.cov_mat_PS = None          # Effective covariance matrix of the (2D marginal of the) post-selected state.

        # Placeholder attributes for arrays that hold Q-function marginal distribution values, FOR PLOTTING PURPOSES ONLY. These should not be used as part of any numerics.
        self.px_Q_PS_values = None        # Array containing post-selection marginal probability distribution values p(X = x).
        self.py_Q_PS_values = None        # Array containing post-selection marginal probability distribution values p(Y = y).
        self.Q_PS_values = None           # Array containing post-selected joint probability distribution values p(X = x, Y = y).

    def plot_Q_marginals(self, normalised_tau_arr, normalised_g_arr, axis_range = [-10, 10], num_points_on_axis = 100, add_originals = False):
        """
            TODO: Redo this to plot p(x) and p_PS(x). NOT the Husimi-Q function marginals. They are used for Holevo calculations only.
            
            Plot the marginal distributions and heatmap of the Husimi-Q function for postselected data.
            This method plots the marginal distributions p_PSQ(x) and p_PSQ(y), as well as the heatmap of the joint distribution p_Q(x, y)
            for postselected data.

            If add_originals is set to True, the original p_Q(x) and p_Q(y) will also be distributed.

            Returns:
                None
        """
        self._evaluate_marginals(normalised_tau_arr, normalised_g_arr, axis_range, num_points_on_axis)

        axis_values = np.linspace(axis_range[0], axis_range[1], num_points_on_axis)
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # Plot px
        axs[0].plot(axis_values, self.px_Q_PS_values, 'k-')
        axs[0].set_title('Plot of px')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('p(x = X)')

        if add_originals:
            # Evaluate the original p_Q(x) for comparison, and normalise
            originals = self.px_rv.pdf(axis_values)
            originals /= np.sum(originals)

            axs[0].plot(axis_values, originals, 'k--')

        # Plot py
        axs[1].plot(axis_values, self.py_Q_PS_values, 'k-')
        axs[1].set_title('Plot of py')
        axs[1].set_xlabel('y')
        axs[1].set_ylabel('p(y = Y)')

        if add_originals:
            # Evaluate the original p(y) for comparison, and normalise
            originals = self.py_rv.pdf(axis_values)
            originals /= np.sum(originals)
            axs[1].plot(axis_values, originals, 'k--')

        # # Plot pxy using imshow, with the range set as axis_range \times axis_range
        # axs[2].imshow(self.Q_PS_values, extent = (axis_values[0], axis_values[-1], axis_values[0], axis_values[-1]), origin = 'lower')
        # axs[2].set_title('p(x = X, y = Y)')

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plot
        plt.show()

    def evaluate_p_pass(self, normalised_tau_arr, normalised_g_arr):
        """
            Evaluate the probability of passing the filter function by integrating over the 2D marginalised version: (F(y) \\times Q*(x, y)).

            Arguments:
                tau_arr: array(float)
                    An array holding the values of $\\tau_i$.
                g_arr: array(float)
                    An array holding the values of $g_{\\pm, i}$. g[i][0] contains $g_{i,-}$ and g[i][1] contains $g_{i,+}$.
        """
        # Scale tau_arr and g_arr by Bob's standard deviation, such that they are in units of Bob's standard deviation.
        tau_arr = np.sqrt(self.bob_variance) * np.array(normalised_tau_arr)
        g_arr = np.sqrt(self.bob_variance) * np.array(normalised_g_arr)

        p_pass = 0.0

        for i in range(len(tau_arr) - 1):
            p_pass += self._integrate_1D_gaussian_pdf(
                self.py_rv, # Bob's random variable, NOT post-selected
                [tau_arr[i] + g_arr[i][1], tau_arr[i + 1] - g_arr[i + 1][0]]
            )

        self.p_pass = p_pass

        return self.p_pass

    def evaluate_cov_mat_PS(self, normalised_tau_arr, normalised_g_arr):
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
        var_x, var_y, cov_xy = self._compute_Q_PS_moments(self.Q_star_rv, normalised_tau_arr, normalised_g_arr)

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
    
    def _evaluate_marginals(self, normalised_tau_arr, normalised_g_arr, axis_range = [-10, 10], num_points_on_axis = 100):
        """
            TODO: Change this to evaluate p(x) and p_PS(x) instead of the Husimi-Q function marginals.

            Evaluate the marginal distributions of Q() for post-selected data.
            This is done by evaluating the marginal distributions p_Q(x) and p_Q(y) for post-selected data.
        """
        # Scale tau_arr and g_arr by Bob's standard deviation, such that they are in units of Bob's standard deviation, as is relevant for evaluating marginals.
        tau_arr = np.sqrt(self.bob_variance) * np.array(normalised_tau_arr)
        g_arr = np.sqrt(self.bob_variance) * np.array(normalised_g_arr)

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
        self.px_Q_PS_values = np.sum(self.Q_PS_values, axis=0)
        self.py_Q_PS_values = np.sum(self.Q_PS_values, axis=1)
        self.px_Q_PS_values /= np.sum(self.px_Q_PS_values)
        self.py_Q_PS_values /= np.sum(self.py_Q_PS_values)

        return self.px_Q_PS_values, self.py_Q_PS_values, self.Q_PS_values

    def _compute_Q_PS_moments(self, rv, normalised_tau_arr, normalised_g_arr):
        """
            Compute the variance and covariance of x and y over specified limits for a 2D Gaussian distribution,
            considering exclusion zones (guard bands).

            Note that this assumes E(x) = E(y) = 0.0. This assumption is valid for the GBSR protcol due to symmetry. This assumption is made to minimise numerical integrations.
        """
        # Scale tau_arr and g_arr by  Bob's standard deviations, such that they are in units of Bob's standard deviation.
        y_tau_arr = np.sqrt(self.bob_variance) * np.array(normalised_tau_arr)
        y_g_arr = np.array(normalised_g_arr) * np.sqrt(self.bob_variance)

        # Low tolerances, as numerics are very expensive
        epsabs  = 1e-3
        epsrel  = 1e-3

        # Define integrands
        def integrand_x2(y, x):
            return x**2 * rv.pdf([x, y])

        def integrand_y2(y, x):
            return y**2 * rv.pdf([x, y])

        def integrand_xy(y, x):
            return x * y * rv.pdf([x, y])

        xlims = [-np.inf, np.inf]
        ylims = [-np.inf, np.inf]

        # Build guard band ranges in absolute terms (in Bob's standard deviation units)
        band_ranges = []
        for i in range(len(tau_arr)):
            band_lower = y_tau_arr[i] - y_g_arr[i][0]
            band_upper = y_tau_arr[i] + y_g_arr[i][1]
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

        # Initialize numerators, to be later normalised by how much of the total distribution is considered
        E_X2_num = 0.0
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

                # Perform numerical integration
                E_X2_num += dblquad(
                    integrand_x2, xlims[0], xlims[1], lambda x: y_start, lambda x: y_end, epsabs=epsabs, epsrel=epsrel
                )[0]

                E_Y2_num += dblquad(
                    integrand_y2, xlims[0], xlims[1], lambda x: y_start, lambda x: y_end, epsabs=epsabs, epsrel=epsrel
                )[0]

                E_XY_num += dblquad(
                    integrand_xy, xlims[0], xlims[1], lambda x: y_start, lambda x: y_end, epsabs=epsabs, epsrel=epsrel
                )[0]

        if norm_const == 0.0:
            raise ValueError("Normalization constant is zero. Check the integration limits and guard bands.")

        # Compute (co)variances and return
        var_x = E_X2_num / norm_const
        var_y = E_Y2_num / norm_const
        cov_xy = E_XY_num / norm_const

        return var_x, var_y, cov_xy

    def _integrate_Q_PS(self, xlims, ylims, normalised_tau_arr, normalised_g_arr):
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
        # Scale tau_arr and g_arr by Bob's standard deviation, such that they are in units of Bob's standard deviation.
        y_tau_arr = np.sqrt(self.bob_variance) * np.array(normalised_tau_arr)
        y_g_arr = np.sqrt(self.bob_variance) * np.array(normalised_g_arr)

        integral_result = 0.0

        y_lower = ylims[0]
        y_upper = ylims[1]

        band_ranges = []
        for i in range(len(tau_arr)):
            band_lower = y_tau_arr[i] - y_g_arr[i][0]
            band_upper = y_tau_arr[i] + y_g_arr[i][1]
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
        # # Extract limits and set to a large magnitude if infinite
        # x1 = xlims[0] if xlims[0] != -np.inf else -1e10
        # x2 = xlims[1] if xlims[1] != np.inf else 1e10
        # y1 = ylims[0] if ylims[0] != -np.inf else -1e10
        # y2 = ylims[1] if ylims[1] != np.inf else 1e10

        # # Compute the probability inside the rectangle defined by the limits
        # return (rv.cdf([x2, y2]) - rv.cdf([x2, y1]) - rv.cdf([x1, y2]) + rv.cdf([x1, y1]))

        result, error = dblquad(lambda x, y: rv.pdf([x, y]), xlims[0], xlims[1], lambda x: ylims[0], lambda x: ylims[1])

        return result

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

        self.gaussian_attack_holevo_information = self._evaluate_holevo_information(self.a, self.b, self.c)

        self.devetak_winter = self.I_AB - self.gaussian_attack_holevo_information

    def plot_guard_band_diagram(self, normalised_tau_arr, normalised_g_arr):
        """
            Plot the guard band diagram for the GBSR protocol.
            This method plots the guard band diagram for the GBSR protocol, showing the guard bands around each tau.
        """
        # Scale tau_arr and g_arr by Bob's standard deviation, such that they are in units of Bob's standard deviation.
        tau_arr = np.sqrt(self.bob_variance) * np.array(normalised_tau_arr)
        g_arr = np.sqrt(self.bob_variance) * np.array(normalised_g_arr)

        fig, ax = plt.subplots(figsize=(2.5 * plt.rcParams["figure.figsize"][0], plt.rcParams["figure.figsize"][1]))

        # Plot a Gaussian of variance self.bob_variance
        x = np.linspace(-2.5*self.py_rv.var(), 2.5*self.py_rv.var(), 200)
        y = self.py_rv.pdf(x)
        ax.plot(x, y, 'r-', label='Bob\'s marginal')

        # Plot the guard bands
        for i in range(len(tau_arr)):
            # Plot interval centre (tau)
            ax.plot([tau_arr[i], tau_arr[i]], [0, 1.1 * max(y)], 'k-')

            # Plot left edge of guard band
            ax.plot([tau_arr[i] - g_arr[i][0], tau_arr[i] - g_arr[i][0]], [0, 1.1 * max(y)], 'k--')

            # Plot right edge of guard band
            ax.plot([tau_arr[i] + g_arr[i][1], tau_arr[i] + g_arr[i][1]], [0, 1.1 * max(y)], 'k--')

        ax.set_xlim([x[0], x[-1]])
        ax.set_ylim([0, 1.2 * max(y)])

        plt.show()

    def evaluate_key_rate_in_bits_per_pulse(
            self,
            tau_arr,
            g_arr,
        ):
        """
            Evaluate the key rate for GBSR with a given number of slices $m$ and an array holding each interval edge, and a 2D array holding the relative guard band spans.

            K_{\infty,\text{PS}} = p_\text{pass} \left(H(S_{1,\cdots,m} (1 - h(e)) - \chi \right)

        """
        quantisation_entropy = self.evaluate_quantisation_entropy(tau_arr)

        error_rate = self.evaluate_error_rate(tau_arr, g_arr)

        classical_leaked_information = self._evaluate_leaked_information(error_rate)
        
        holevo_information = self._evaluate_holevo_information(self.a, self.b, self.c)

        p_pass = self.evaluate_p_pass(tau_arr, g_arr)

        return p_pass * (quantisation_entropy - classical_leaked_information - holevo_information)

    def evaluate_error_rate(self, normalised_tau_arr, normalised_g_arr, bit_assignment = "Gray"):
        """
            Evaluate the error rate of the protocol. Note that, of course, this operates on the statistics BEFORE post-selection.

            e = \sum_{i = 0}^{2^m - 1} \sum_{j = 0, j \neq i}^{2^m - 1} \: \int_{\tau_i}^{\tau_{i+1}} \! dX \int_{\tau_j + g_{j,+}}^{\tau_{j+1} - g_{j+1,-}} \! dY \: \frac{d_H(\mathbf{b}_i, \mathbf{b}_j)}{m} p(X, Y) \\
        """

        # As tau arr and g arr are passed as normalised, we need to scale them for the appropriate limits.
        # For xlims, this is tantamount to scaling by Alice's standard deviation.
        # For ylims, scale by Bob's standard deviation.
        # The same can be applied to normalised_g_arr
        x_tau_arr = np.sqrt(self.alice_variance) * np.array(normalised_tau_arr)
        y_tau_arr = np.sqrt(self.bob_variance) * np.array(normalised_tau_arr)
        y_g_arr = np.array(normalised_g_arr) * np.sqrt(self.bob_variance)

        # Next, generate the bit assignment array
        if bit_assignment == "Gray":
            bit_strings = self._generate_gray_bit_assignment(self.m)
        elif bit_assignment == "binary":
            bit_strings = self._generate_binary_bit_assignment(self.m)
        else:
            raise ValueError("Invalid bit assignment scheme. Please choose 'Gray' or 'binary' or implement another.")

        error_rate = 0.0

        for i in range(self.number_of_intervals):
            for j in range(self.number_of_intervals):
                if i == j:
                    continue

                # Calculate the normalised Hamming distance between the bit strings
                normalised_Hamming_distance = self._hamming_distance(bit_strings[i], bit_strings[j]) / self.m

                # Define the limits of integration for X and Y
                xlims = [x_tau_arr[i], x_tau_arr[i + 1]]
                ylims = [y_tau_arr[j] + y_g_arr[j][1], y_tau_arr[j + 1] - y_g_arr[j + 1][0]]

                # Integrate the joint PDF over the given limits, and multiply by the normalised Hamming distance
                error_rate += normalised_Hamming_distance * self._integrate_2D_gaussian_pdf(self.joint_rv, xlims, ylims)

        return error_rate

    def evaluate_quantisation_entropy(self, normalised_tau_arr):
        """
            Evaluate the quantisation entropy of the protocol.

            H = - \\sum_{i = 0}^{2^m - 1} \\int_{\tau_i}^{\tau_{i+1}} \\! dX \\: p(X = x) \\log_2 p(X = x)
        """
        # With normalised tau arr, scale using the standard deviation of Alice's random variable (self.px_rv), to retrieve tau_arr in units of Alice's standard deviation
        tau_arr = [normalised_tau_arr[i] * np.sqrt(self.alice_variance) for i in range(len(normalised_tau_arr))]

        interval_probabilities = [self._integrate_1D_gaussian_pdf(self.px_rv, [tau_arr[i], tau_arr[i+1]]) for i in range(self.number_of_intervals)]
        
        # Remove 0.0 probabilities, as they will cause the entropy to be NaN, although in this limit we want 0.0.
        interval_probabilities = [p for p in interval_probabilities if p != 0.0]

        return -1.0 * np.sum([interval_probabilities[i] * np.log2(interval_probabilities[i]) for i in range(self.number_of_intervals)])

    def _evaluate_mutual_information(self):
        """
            Evaluate the mutual information between Alice and Bob's correlated Gaussian variables.
            This will later be done via numerical integration but the known analytical form can be read off for now.
        """
        pass

    def _evaluate_leaked_information(self, error_rate):
        return self.m * self._binary_entropy(error_rate)

    def _evaluate_slepian_wolf_leaked_information(self, normalised_tau_arr, normalised_g_arr):
        """
        Calculate a lower bound on the amount of information leaked on the public channel, using the Slepian-Wolf theorem.

        LEAK = H(S_{1, ..., m}(X) | Y)
            = H(S_{1, ..., m}(X), Y) - H(Y)
        """

        # Step 1: Discretize y
        num_y_points = 1000
        sigma_Y = np.sqrt(self.bob_variance)
        y_min = -5 * sigma_Y
        y_max = 5 * sigma_Y
        y_vals = np.linspace(y_min, y_max, num_y_points)
        delta_y = y_vals[1] - y_vals[0]

        # Step 2: Compute p(y)
        p_y = np.array([self._compute_p_y(y_j) for y_j in y_vals])
        epsilon = 1e-12
        p_y = np.clip(p_y, epsilon, None)

        # Step 3: Compute p(s_i, y)
        num_intervals = self.number_of_intervals
        p_s_y = np.zeros((num_intervals, num_y_points))

        x_tau_arr = np.sqrt(self.alice_variance) * np.array(normalised_tau_arr)

        for i in range(num_intervals):
            xlims = [x_tau_arr[i], x_tau_arr[i+1]]
            for idx, y_j in enumerate(y_vals):
                p_s_y[i, idx] = self._integrate_pxy_over_x(xlims, y_j)

        # Step 4: Compute p(s_i | y)
        p_s_given_y = p_s_y / p_y

        # Step 5: Compute H(S | Y = y_j)
        H_S_given_y = np.zeros(num_y_points)

        for idx in range(num_y_points):
            p_s_given_y_j = p_s_given_y[:, idx]
            p_s_given_y_j = np.clip(p_s_given_y_j, epsilon, None)
            H_S_given_y[idx] = -np.sum(p_s_given_y_j * np.log2(p_s_given_y_j))

        # Step 6: Compute H(S|Y)
        H_S_given_Y = np.sum(H_S_given_y * p_y) * delta_y

        return H_S_given_Y
    
    def _evaluate_holevo_information(self, a, b, c):
        """
            An upper bound for the Holevo information using the calculated covariance matrix. 
            See first-year report or (Laudenbach 2018) for more details.
            We can directly find the symplectic eigenvalues of the needed covariance matrix by using the familiar formulae.
        """
        nu = np.zeros(3)

        sqrt_value = (a + b)**2 - (4.0 * c**2)

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
        """
            Calculate the binary entropy of a given error rate.
        """
        if not 0.0 <= e <= 1.0:
            raise ValueError("Error rate e must be between 0 and 1 inclusive.")

        epsilon = 1e-15
        e = min(max(e, epsilon), 1 - epsilon)

        one_minus_e = 1.0 - e
        return - (e * np.log2(e)) - (one_minus_e * np.log2(one_minus_e))

    def _generate_gray_bit_assignment(self, m):
        """
            Generate the Gray bit assignment for the GBSR protocol.
            Return an array of bit strings from 0 to 2^m - 1.
        """
        bit_strings = []

        for i in range(2**m):
            # The ith gray code is generated using the formula i ^ (i >> 1)
            gray_code = i ^ (i >> 1)
            # Format the gray code into a bit string with 'm' bits
            bit_strings.append(f'{gray_code:0{m}b}')

        return bit_strings
    
    def _generate_binary_bit_assignment(self, m):
        """
            Generate the binary bit assignment for the GBSR protocol.
            Return an array of bit strings from 0 to 2^m - 1.
        """
        bit_strings = []

        for i in range(2**m):
            # Format the number 'i' as a binary string with 'm' bits
            bit_strings.append(f'{i:0{m}b}')

        return bit_strings
    
    def _hamming_distance(self, bit_string_1, bit_string_2):
        """
            Calculate the Hamming distance between two bit strings.
        """
        return sum([bit_string_1[i] != bit_string_2[i] for i in range(len(bit_string_1))])

if __name__ == "__main__":
    # Example usage

    gbsr = GBSR(1, 2.2, 0.4, 0.05, progress_bar=True)

    tau_arr = [-np.inf, 0.0, np.inf]
    g_arr = [
        [0.0, 0.0],
        [0.3, 0.3],
        [0.0, 0.0]
    ]

    gbsr.plot_Q_marginals(tau_arr, g_arr)

    