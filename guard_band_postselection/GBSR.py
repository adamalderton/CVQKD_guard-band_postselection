import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from numba import njit, vectorize, guvectorize, float64
from scipy.integrate import nquad
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal
import time

class GBSR_quantum_statistics():
    def __init__(
            self,
            modulation_variance,
            transmittance,
            excess_noise,
            grid_range = [-10, 10],
            num_points_on_axis = 64,
            JIT = True
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

        self.alice_variance = modulation_variance + 1.0                              # Alice's effective variance $V_A = $V_\\text{mod} + 1$ in SNU.
        self.bob_variance = (transmittance * modulation_variance) + 1 + excess_noise # Bob's effective variance $V_B = T V_\\text{mod} + 1 + \\xi$ in SNU.

        # Covariance matrix coefficients derived from the above
        self.a = self.alice_variance
        self.b = self.bob_variance
        self.c = np.sqrt(self.transmittance * (self.alice_variance*self.alice_variance - 1))

        # Initialise covariance matrix and its coefficients a, b and c using sympy. Numerical values can be substituted in later.
        self.a_sym, self.b_sym, self.c_sym = sp.symbols('a b c', real = True, positive = True)
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

        # Define the Husimi-Q function of a TMSV vacuum state, subject to no post-selection, in sympy.
        # self.Q_star_sym = (sp.sqrt(self.gamma_mat.det()) / sp.pi**2) * sp.exp((-self.a_sym * (sp.Abs(self.alpha_sym))**2) - (self.b_sym * (sp.Abs(self.beta_sym))**2) - (2 * self.c_sym * sp.Abs(self.alpha_sym) * sp.Abs(self.beta_sym) * sp.cos(sp.arg(self.alpha_sym) - sp.arg(self.beta_sym))))
        # x = sp.Matrix([self.alpha_re_sym, self.alpha_im_sym, self.beta_re_sym, self.beta_im_sym])
        # self.Q_star_sym = (sp.sqrt(self.gamma_mat.det()) / sp.pi**2) * sp.exp(-1 * x.T * self.gamma_mat * x)[0]
        # self.Q_star_sym = (1 / (sp.pi**2 * sp.sqrt(sp.det(self.cov_mat_sym + sp.eye(4))))) * sp.exp(-1 * x.T * sp.Inverse(self.cov_mat_sym + sp.eye(4)) * x)[0]

        # Generate the Q_star lambda function with, a, b and c as parameters. This must be updated when a, b and c change.
        # self.Q_star_lambda = self._generate_Q_star_lambda()
        self.Q_star_lambda = multivariate_normal(
            mean = np.zeros(4),
            cov = self.cov_mat_sym.subs({self.a_sym: self.a, self.b_sym: self.b, self.c_sym: self.c}) + np.eye(4)
        ).pdf

        # Generate the grid over which to perform necessary numerical integrations.
        self.num_points_on_axis = num_points_on_axis
        self.axis_values = np.linspace(grid_range[0], grid_range[1], self.num_points_on_axis)
        self.alpha_re_mesh, self.alpha_im_mesh, self.beta_re_mesh, self.beta_im_mesh = np.meshgrid(self.axis_values, self.axis_values, self.axis_values, self.axis_values, indexing = "ij")

        # Generate Q_star_values. These are constant for now and do not need to be updated (for constant a, b and c)
        #self.Q_star_values = self.Q_star_lambda(self.alpha_re_mesh, self.alpha_im_mesh, self.beta_re_mesh, self.beta_im_mesh)
        self.Q_star_values = self.Q_star_lambda(np.stack((self.alpha_re_mesh, self.alpha_im_mesh, self.beta_re_mesh, self.beta_im_mesh), axis = -1))
        self.Q_star_values /= np.sum(self.Q_star_values)

        # Placeholder attributes for those that need to be evaluated with the specifics of the guard bands in mind.
        self.F_sym = None           # Symbolic representation of the filter function $F(\beta)$.
        self.p_pass = None          # Numerical value of the probability of passing the filter function.
        self.Q_PS_values = None     # Array of values of the Q function AFTER post-selection on a grid of points.
        self.marginals = []         # List of marginal probability distributions for alpha_re, alpha_im, beta_re and beta_im.
        self.marginal_means = []    # List of means for the marginal probability distributions for alpha_re, alpha_im, beta_re and beta_im.
        self.px = None              # Array containing marginal probability distribution values p(X = x).
        self.py = None              # Array containing marginal probability distribution values p(Y = y).
        self.effective_cov_mat = None # Effective covariance matrix of the post-selected state.
        self.a_PS = None            # Effective covariance matrix coefficient a_PS.
        self.b_PS = None            # Effective covariance matrix coefficient b_PS.
        self.c_PS = None            # Effective covariance matrix coefficient c_PS. 

    def plot_marginals(self):
        """
            Plot the marginal probability distributions p(X = x), p(Y = y) and p(X = x, Y = y) using the joint probability distribution p(alpha_re, alpha_im, beta_re, beta_im).
            This is set up to work in iPython environments, such as Jupyter notebooks.
        """
        # Set the figure size
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Plot px
        axs[0].plot(self.axis_values, self.px)
        axs[0].set_title('Plot of px')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('px')

        # Plot py
        axs[1].plot(self.axis_values, self.py)
        axs[1].set_title('Plot of py')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('py')

        # Plot pxy using imshow, with the range set as axis_range \times axis_range
        # First, calculate a pxy marginal via techniques used in self._evaluate_effective_covariance_matrix(). That is, integrate over the imaginary parts.
        pxy = np.sum(self.Q_PS_values, axis = (1, 3))
        pxy /= np.sum(pxy)
        axs[2].imshow(pxy, extent = (self.axis_values[0], self.axis_values[-1], self.axis_values[0], self.axis_values[-1]), origin='lower')
        axs[2].set_title('Heatmap of pxy')
        axs[2].set_xlabel('Index')
        axs[2].set_ylabel('Index')

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plot
        plt.show()

    def _evaluate_p_pass_and_Q_PS_values(self, tau_arr, g_arr):
        """
        Evaluate the probability of passing the filter function by integrating over (F(\beta) \times Q*(\alpha, \beta)) over the entire complex plane.

        Arguments:
            tau_arr: array(float)
                An array holding the values of $\tau_i$.
            g_arr: array(float)
                An array holding the values of $g_{\pm, i}$. g[i][0] contains $g_{i,-}$ and g[i][1] contains $g_{i,+}$.
        """
        F_lambda = self._generate_F_lambda(tau_arr, g_arr)

        # Find F * Q_star, in numerical form.
        # We store the intermediate (unnormalised) Q_PS_values for later use in self.QS_PS_values for now.
        # These values will shortly be normalised by division by p_pass.
        self.Q_PS_values = F_lambda(self.beta_re_mesh, self.beta_im_mesh) * self.Q_star_values

        # Integrate over the entire complex plane (using a 4D trapz) to find p_pass
        self.p_pass = self._4D_integral_over_Q(self.Q_PS_values)

        # Divide through by p_pass to find normalise Q_PS_values
        self.Q_PS_values /= self.p_pass

        return self.p_pass, self.Q_PS_values

    def _generate_F_lambda(self, tau_arr, g_arr):
        """
        Substitute the guard band properties into the filter function $F(\beta)$.

                Arguments:
            F_sym: sympy expression
                The filter function $F(\beta)$.
            tau_arr: array(float)
                An array holding the values of $\tau_i$.
            g_arr: array(float)
                An array holding the values of $g_{\pm, i}$. g[i][0] contains $g_{i,-}$ and g[i][1] contains $g_{i,+}$.
        """
        return sp.lambdify(
            (self.beta_re_sym, self.beta_im_sym),
            self.F_sym.subs(
                [(sp.Symbol(f'g_minus{i}'), g_arr[i][0]) for i in range(len(g_arr))] +
                [(sp.Symbol(f'g_plus{i}'), g_arr[i][1]) for i in range(len(g_arr))] +
                [(sp.Symbol(f'taus{i}'), tau_arr[i]) for i in range(len(tau_arr))] +
                [(self.beta_sym, self.beta_re_sym + self.beta_im_sym*1j)]
            ),
            "numpy"
        )

    def _define_filter_function(self, m):
        """
        Define the filter function $F(\beta)$ in sympy. This is done in a standalone function as it is quite an involved process.

        F(\beta) = 
            \begin{cases} 
                0 & \text{if} \ \exists x \in \{\beta_\text{re}, \beta_\text{im}\} \ \text{such that} \\
                  & \quad \forall i \in \{0, 1, \ldots, |\mathcal{T}|\}, g_{-,i} \leq (x - \tau_i) \leq g_{+,i} \\
                1 & \text{otherwise}
            \end{cases}

        Arguments:
            m: Integer.
                The number of slices $m$. This produces a number of intervals $|T| = 2^m$, therefore there is $|T| + 1$ interval edges.
        """
        T = 2**m

        # Define the sympy symbols. See paper for thorough definitions

        g_minus = sp.symbols(f'g_minus:{T+1}')  # Creates g_minus_0, g_minus_1, ..., g_minus_T
        g_plus = sp.symbols(f'g_plus:{T+1}')    # Creates g_plus_0, g_plus_1, ..., g_plus_T
        taus = sp.symbols(f'taus:{T+1}')        # Creates taus_0, taus_1, ..., taus_T

        # Construct the conditions
        conditions = []
        for i in range(T + 1):
            conditions.append(sp.And((-1 * g_minus[i]) <= (sp.re(self.beta_sym) - taus[i]), (sp.re(self.beta_sym) - taus[i]) <= g_plus[i]))
            conditions.append(sp.And((-1 * g_minus[i]) <= (sp.im(self.beta_sym) - taus[i]), (sp.im(self.beta_sym) - taus[i]) <= g_plus[i]))
        
        # Define the filter function. This unpacks conditions into essentially a huge OR statement.
        self.F_sym = sp.Piecewise(
            (0, sp.Or(*conditions)),
            (1, True)
        )

        return self.F_sym

    def _generate_Q_star_lambda(self):
        """
            Generate the Q function (subject to no post-selection, hence Q_star) for a given covariance matrix in a lambda function, that is JIT compiled.
            IMPORTANT: Due to needing real integrands for scipy.integrate.nquad, the Q function is evaluated with the real and imaginary parts of alpha and beta as separate arguments.
            That is, Q_star_lambda(alpha_re, alpha_im, beta_re, beta_im).
        """
        self.Q_star_lambda = sp.lambdify(
            (self.alpha_re_sym, self.alpha_im_sym, self.beta_re_sym, self.beta_im_sym), # Function arguments
            self.Q_star_sym.subs(
                [
                    (self.alpha_sym, self.alpha_re_sym + self.alpha_im_sym*1j), # alpha = alpha_re + i alpha_im
                    (self.beta_sym, self.beta_re_sym + self.beta_im_sym*1j),    # beta = beta_re + i beta_im
                    (self.a_sym, self.a),
                    (self.b_sym, self.b),
                    (self.c_sym, self.c)]
            ).simplify(),
        )

        if self.JIT:
            self.Q_star_lambda = vectorize([float64(float64, float64, float64, float64)], target = "parallel")(self.Q_star_lambda)

        return self.Q_star_lambda

    def _evaluate_Q_PS_marginals(self):
        """
            Evaluate the marginal probability distributions p(X = x), p(Y = y) and p(X = x, Y = y) using the joint probability distribution p(alpha_re, alpha_im, beta_re, beta_im).
        """
        alpha_re_marginal = np.sum(self.Q_PS_values, axis = (1, 2, 3))
        alpha_im_marginal = np.sum(self.Q_PS_values, axis = (0, 2, 3))
        beta_re_marginal = np.sum(self.Q_PS_values, axis = (0, 1, 3))
        beta_im_marginal = np.sum(self.Q_PS_values, axis = (0, 1, 2))

        # Normalise
        alpha_re_marginal /= np.sum(alpha_re_marginal)
        alpha_im_marginal /= np.sum(alpha_im_marginal)
        beta_re_marginal /= np.sum(beta_re_marginal)
        beta_im_marginal /= np.sum(beta_im_marginal)

        # Assign marginals to px and py. For now, we choose the real elements.
        self.px = alpha_re_marginal
        self.py = beta_re_marginal

        # Store resulting values
        self.marginals = [alpha_re_marginal, alpha_im_marginal, beta_re_marginal, beta_im_marginal]
        self.marginal_means = [np.sum(self.marginals[i] * self.axis_values) for i in range(4)]

        return self.marginals

    def _evaluate_effective_covariance_matrix(self):
        """
            Evaluate the effective covariance matrix of the post-selected state via finding (co)variances of the appropriate marginals.

        """
        effective_husimi_cov_mat = np.zeros((4, 4))

        # Initialise and calculate marginals (and their means).
        self._evaluate_Q_PS_marginals()
        joint_marginal = np.zeros((self.num_points_on_axis, self.num_points_on_axis))
        
        for i in range(4):
            for j in range(4):
                if j == i:
                    # Leading diagonal elements are just variances of marginals
                    effective_husimi_cov_mat[i][i] = np.sum((self.axis_values - self.marginal_means[i])**2 * self.marginals[i])
                    continue

                # Off-diagonal elements are covariances of marginals. First, calculate joint marginal and normalise.
                joint_marginal = np.sum(self.Q_PS_values, axis = tuple(k for k in [0, 1, 2, 3] if k not in [i, j]))
                joint_marginal /= np.sum(joint_marginal)

                # Use joint marginal to calculate covariance
                effective_husimi_cov_mat[i][j] = np.sum(joint_marginal * np.outer(self.axis_values - self.marginal_means[i], self.axis_values - self.marginal_means[j])) - (self.marginal_means[i] * self.marginal_means[j])
        
        # The covariance matrix of the state is therefore the effective husimi covariance matrix minus the identity matrix.
        self.effective_cov_mat = effective_husimi_cov_mat - np.eye(4)

        return self.effective_cov_mat

    def _evaluate_a_PS_b_PS_c_PS(self):
        """
            Evaluate the effective covariance matrix coefficients for the given post-selected
            state represented in Q_PS_values.
        """
        self.effective_cov_mat = self._evaluate_effective_covariance_matrix()

        self.a_PS = self.effective_cov_mat[0][0]
        self.b_PS = self.effective_cov_mat[2][2]
        self.c_PS = self.effective_cov_mat[0][2]

        return self.a_PS, self.b_PS, self.c_PS

    def _4D_integral_over_Q(self, integrand_values):
        # Perform summation over each axis using np.sum
        return np.sum(integrand_values)

class GBSR(GBSR_quantum_statistics):
    def __init__(
            self,
            m,
            modulation_variance,
            transmittance,
            excess_noise
        ) -> None:
        super().__init__(modulation_variance, transmittance, excess_noise)

        self.m = m

        # Define the symbolic filter function
        self.F_sym = self._define_filter_function(m)

        # Need to evaluate certain functions. (Assuming m = 1 for now).
        # Substitute toy guard band for now.
        self.tau_arr = np.array([-np.inf, 0.0, np.inf])
        self.g_arr = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

        # Evaluate the probability of passing the filter function and the Q function after post-selection.
        self.p_pass, self.Q_PS_values = self._evaluate_p_pass_and_Q_PS_values(self.tau_arr, self.g_arr)

        # Evaluate marginals for now.
        self._evaluate_Q_PS_marginals()
        
    def evaluate_key_rate_in_bits_per_pulse(self, m, tau_arr, g_arr):
        """
            Evaluate the key rate for GBSR with a given number of slices $m$ and an array holding each interval edge, and a 2D array holding 

            Arguments:
                m: integer
                    The number of slices.
                interval_edges: array(float)
                    An array holding the edges of each interval, from left to right. 
                    That is, the first interval is -np.inf and the last is np.inf.
                    Of course, the non-extremal values of the array should be finite and in ascending order.
                    The interval edges should be given in units of standard deviation. That is, an interval which is one standard deviation from the mean would have a value of 1.0.
                gb_widths: array(array(float))
                    A 2D array holding the widths of the guard bands for each interval.
                    The first index corresponds to the negative extent from the interval edge, and the second index corresponds to the positive extent.
                    
            Returns:
                key_rate: float
                    The key rate for the given number of slices and interval edge positions.
        """
        pass
        #quantisation_entropy = integrator.evaluate_slicing_entropy(m, interval_edges)
        #leaked_classical_information = integrator.evaluate_error_correction_information(m, interval_edges)

        #return ((quantisation_entropy - leaked_classical_information) / self.mutual_information) * (self.mutual_information - self.holevo_information)

    def _evaluate_mutual_information(self):
        """
            Evaluate the mutual information between Alice and Bob's probability distributions.
            This will later be done via numerical integration but the known analytical form can be read off for now.
        """
        snr = (self.T * self.v_mod) / (1.0 + self.xi)
        return 0.5 * np.log2(1.0 + snr)

    def _evaluate_holevo_information(self):
        """
            An upper bound for the Holevo information using the calcuated covariance matrix. See first year report or (Laudenbach 2018) for more details.
            We can directly find the symplectic eigenvalues of the needed covariance matrix by using the familiar formulae.
            Here we initialise all symplectic eigenvalues, represented by an array named nu. The last element is evaluated later.
        """
        nu = np.zeros(3)
        nu[0] = 0.5 * (np.sqrt((self._a + self._b)**2 - 4.0*self._c*self._c) + (self._b - self._a))
        nu[1] = 0.5 * (np.sqrt((self._a + self._b)**2 - 4.0*self._c*self._c) - (self._b - self._a))

        # Now, we need to find the symplectic eigenvalue of the covariance matrix describing A conditioned on B.
        # Following the formula in the notes:
        nu[2] = self._a - ( (self._c*self._c) / (self._b + 1) )

        # With all the necessary symplectic eigenvalues, we can now find the Holevo information:
        return self._g(nu[0]) + self._g(nu[1]) - self._g(nu[2])
    
    def _g(self, x):
        """
            Needed to find the correct contrubution to the Holevo information for each symplectic eigenvalue from the appropriate covariance matrix.
        """
        return ((x + 1.0) / 2.0) * np.log2((x + 1.0) / 2.0) - ((x - 1.0) / 2.0) * np.log2((x - 1.0) / 2.0)
    
if __name__ == "__main__":
    from pprint import pprint
    import numpy as np
    import sympy as sp
    import seaborn as sb
    import matplotlib.pyplot as plt

    gbsr = GBSR(1, 1.0, 1.0, 0.0)

    