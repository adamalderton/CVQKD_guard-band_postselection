import numpy as np
import sympy as sp
from numba import njit, vectorize, float64
from scipy.integrate import nquad
from scipy.signal import convolve
import time

class GBSR_quantum_statistics():
    def __init__(
            self,
            modulation_variance,
            transmittance,
            excess_noise,
            grid_range = [-6, 6],
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

        self.cov_mat = sp.Matrix([
            [self.a_sym, 0, self.c_sym, 0],
            [0, self.a_sym, 0, -self.c_sym],
            [self.c_sym, 0, self.b_sym, 0],
            [0, -self.c_sym, 0, self.b_sym]
        ])
    
        # Define symbols for Alice and Bob's coherent states.
        self.alpha_sym, self.beta_sym = sp.symbols('alpha beta', complex = True)
        
        # Define symbols for the real and complex parts of Alice and Bob's coherent states. These correspond to quadratures q and p.
        self.q_1_sym, self.p_1_sym, self.q_2_sym, self.p_2_sym = sp.symbols('q_1 p_1 q_2 p_2', real = True)

        # Store vectors in symbolic vector x
        self.x = sp.Matrix([self.q_1_sym, self.p_1_sym, self.q_2_sym, self.p_2_sym])

        # Define the Wigner function of a TMSV vacuum state, subject to no post-selection, in sympy. "Star" implies no post-selection.
        self.W_star_sym = (1 / (4 * sp.pi**2 * sp.sqrt(sp.det(self.cov_mat)))) * sp.exp(sp.Rational(-1, 2) * (self.x.T * self.cov_mat.inv() * self.x))

        # Generate the Wigner_star lambda function with a, b and c as parameters.
        self.W_star_lambda = self._generate_W_star_lambda()

        # Generate the grid over which to perform necessary numerical integrations.
        self.axis_range = np.linspace(grid_range[0], grid_range[1], num_points_on_axis)
        self.q_1_mesh, self.p_1_mesh, self.q_2_mesh, self.p_2_mesh = np.meshgrid(self.axis_range, self.axis_range, self.axis_range, self.axis_range)

        # Perform Weierstrass transform to find Husimi-Q function values for given covariance matrix a, b and c.
        self.Q_star_values = self._generate_Q_star_values()

        # Placeholder attributes for those that need to be evaluated with the specifics of the guard bands in mind.
        self.F_sym = None           # Symbolic representation of the filter function $F(\beta)$.
        self.p_pass = None          # Numerical value of the probability of passing the filter function.
        self.Q_PS_values = None     # Array of values of the Q function AFTER post-selection on a grid of points.
        self.px = None              # Array containing marginal probability distribution values p(X = x).
        self.py = None              # Array containing marginal probability distribution values p(Y = y).
        self.pxy = None             # 2D array containing joint probability distribution values p(X = x, Y = y).
        self.eff_cov_mat = None     # Effective covariance matrix for the post-selected state.
        self.a_PS = None            # Effective covariance matrix coefficient a_PS.
        self.b_PS = None            # Effective covariance matrix coefficient b_PS.
        self.c_PS = None            # Effective covariance matrix coefficient c_PS. 

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
        self.Q_PS_values = F_lambda(self.q_2_mesh, self.p_2_mesh) * self.Q_star_values

        # Integrate over the entire complex plane (using a 4D trapz) to find p_pass
        self.p_pass = self._4D_trapz_over_Q(self.Q_PS_values)

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
            (self.q_2_sym, self.p_2_sym),
            self.F_sym.subs(
                [(sp.Symbol(f'g_minus{i}'), g_arr[i][0]) for i in range(len(g_arr))] +
                [(sp.Symbol(f'g_plus{i}'), g_arr[i][1]) for i in range(len(g_arr))] +
                [(sp.Symbol(f'taus{i}'), tau_arr[i]) for i in range(len(tau_arr))] +
                [(self.beta_sym, self.q_2_sym + self.p_2_sym*1j)]
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

    def _generate_W_star_lambda(self):
        """
            Generate the Wigner function for a TMSV state, subject to no post-selection, in a lambda function that is JIT compiled.
            IMPORTANT: Due to needing real integrands for scipy.integrate.nquad, the Wigner function is evaluated with the real and imaginary parts of alpha and beta as separate arguments.
            That is, W_star_lambda(alpha_re, alpha_im, beta_re, beta_im).
        """
        self.W_star_lambda = sp.lambdify(
            (self.q_1_sym, self.p_1_sym, self.q_2_sym, self.p_2_sym), # Function arguments
            self.W_star_sym.subs(
                [
                    (self.alpha_sym, self.q_1_sym + self.p_1_sym*1j), # alpha = q_1 + i*p_1
                    (self.beta_sym, self.q_2_sym + self.p_2_sym*1j), # beta = q_2 + i*p_2
                    (self.a_sym, self.a),
                    (self.b_sym, self.b),
                    (self.c_sym, self.c)
                ]
            ).simplify(),
        )

        if self.JIT:
            self.W_star_lambda = vectorize([float64(float64, float64, float64, float64)], target = "parallel")(self.W_star_lambda)
        
        return self.W_star_lambda
    
    def _generate_Q_star_values(self):
        """
            Generate Wigner function values across the meshgrid, then perform the Weierstrass transform to find the Husimi-Q function values for the given covariance matrix a, b and c.
        """

        # Evaluate the Wigner function values across the meshgrid. Using np.squeeze to remove singleton dimensions.
        W_star_values = np.squeeze(self.W_star_lambda(self.q_1_mesh, self.p_1_mesh, self.q_2_mesh, self.p_2_mesh))

        # Evaluate the Gaussian kernel values across the meshgrid.
        kernel_values = (1 / np.pi) * np.exp(-1.0 * (self.q_1_mesh**2 + self.p_1_mesh**2 + self.q_2_mesh**2 + self.p_2_mesh**2))

        # Perform the appropriate FFTs to find the Husimi-Q function values
        self.Q_star_values = convolve(W_star_values, kernel_values)

        return self.Q_star_values

    def _evaluate_Q_PS_marginals(self):
        """
            Evaluate the marginal probability distributions p(X = x), p(Y = y) and p(X = x, Y = y) using the joint probability distribution p(p_1, q_1, p_2, q_2).
        """
        # Integrate out alpha_im and beta_im which correspond to axis = 3 and axis = 1 respectively.
        self.pxy = np.trapz(np.trapz(self.Q_PS_values, self.axis_range, axis = 3), self.axis_range, axis = 1)

        # Integrate out beta_re, which is axis = 1 for pxy.
        self.px = np.trapz(self.pxy, self.axis_range, axis = 1)

        # Integrate out alpha_re, which is axis = 0 for pxy.
        self.py = np.trapz(self.pxy, self.axis_range, axis = 0)

        return self.px, self.py, self.pxy

    def _evaluate_a_PS_b_PS_c_PS(self):
        """
            Evaluate the effective covariance matrix coefficients for the given post-selected
            state represented in Q_PS_values.
        """
        if self.eff_cov_mat is None:
            self._evaluate_effective_covariance_matrix()
        
        # Read out a_PS, b_PS and c_PS from the appropriate elements of effective covariance matrix.
        self.a_PS = self.eff_cov_mat[0][0]
        self.b_PS = self.eff_cov_mat[2][2]
        self.c_PS = self.eff_cov_mat[0][2]

        return self.a_PS, self.b_PS, self.c_PS

    def _4D_trapz_over_Q(self, integrand_values):
        # Perform integration over each axis using np.trapz
        integral = np.trapz(integrand_values, x = self.axis_range)
        integral = np.trapz(integral, x = self.axis_range)
        integral = np.trapz(integral, x = self.axis_range)
        integral = np.trapz(integral, x = self.axis_range)

        return integral

    def _evaluate_effective_covariance_matrix(self):
        """
            For now, brute force all the 4x4 matrix elements of the covariance matrix.
            Going to do the full 4D integration for now so as to not assume real \equiv im invariance.
        """
        self.eff_cov_mat = np.zeros((4, 4))

        mesh_vals = [self.q_1_mesh, self.p_1_mesh, self.q_2_mesh, self.p_2_mesh]

        for i, vals_i in enumerate(mesh_vals):
            for j, vals_j in enumerate(mesh_vals):
                self.eff_cov_mat[i][j] = self._4D_trapz_over_Q(vals_i * vals_j * self.Q_PS_values)
        

class GBSR(GBSR_quantum_statistics):
    def __init__(
            self,
            m,
            modulation_variance,
            transmittance,
            excess_noise,
            JIT = True
        ) -> None:
        super().__init__(modulation_variance, transmittance, excess_noise, JIT = JIT)

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
        self.px, self.py, self.pxy = self._evaluate_Q_PS_marginals()
        
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

    gbsr = GBSR(1, 1.0, 1.0, 0.0, JIT = False)