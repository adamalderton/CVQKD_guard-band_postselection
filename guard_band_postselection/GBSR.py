import numpy as np
import sympy as sp
from numba import njit, prange
from scipy.integrate import nquad
import time

class GBSR_quantum_statistics():
    def __init__(self, modulation_variance, transmittance, excess_noise, NJIT = True) -> None:
        """
        Class arguments:
            modulation_variance: Alice's modulation variance $V_\\text{mod}$.
            transmittance: Transmissivity $T$ of the channel.
            excess_noise: Excess noise $\\xi$ of the channel.
        """

        self.modulation_variance = modulation_variance
        self.transmittance = transmittance
        self.excess_noise = excess_noise

        self.NJIT = NJIT

        self.alice_variance = modulation_variance + 1.0                              # Alice's effective variance $V_A = $V_\\text{mod} + 1$ in SNU.
        self.bob_variance = (transmittance * modulation_variance) + 1 + excess_noise # Bob's effective variance $V_B = T V_\\text{mod} + 1 + \\xi$ in SNU.

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
        
        # Define symbols for the real and complex parts of Alice and Bob's coherent states.
        self.alpha_re_sym, self.alpha_im_sym = sp.symbols('alpha_re alpha_im', real = True)
        self.beta_re_sym, self.beta_im_sym = sp.symbols('beta_re beta_im', real = True)

        # Define the Husimi-Q function of a TMSV vacuum state, subject to no post-selection, in sympy.
        self.Q_star_sym = (sp.sqrt(self.cov_mat.det()) / sp.pi**2) * sp.exp(-self.a_sym * (sp.Abs(self.alpha_sym))**2 - self.b_sym * sp.Abs(self.beta_sym)**2 - 2 * self.c_sym * sp.Abs(self.alpha_sym) * sp.Abs(self.beta_sym) * sp.cos(sp.arg(self.alpha_sym) - sp.arg(self.beta_sym)))

        # Generate the Q_star lambda function with, a, b and c as parameters. This must be updated when a, b and c change.
        # This should probably be done as a class property but it's likely just me using this code so this is probably okay.
        self.Q_star_lambda = self._generate_Q_star_lambda(
            self.alice_variance,
            self.bob_variance,
            np.sqrt(self.transmittance * (self.alice_variance*self.alice_variance - 1))
        )

        # Placeholder attributes for those that need to be evaluated with the specifics of the guard bands in mind.
        self.F_sym = None       # Symbolic representation of the filter function $F(\beta)$.
        self.p_pass = None      # Numerical value of the probability of passing the filter function.
        self.Q_PS_lambda = None # Lambda function for the Q function subject to post-selection.
        self.pX = None          # Array containing marginal probability distribution values p(X = x).
        self.pY = None          # Array containing marginal probability distribution values p(Y = y).
        self.pXY = None         # 2D array containing joint probability distribution values p(X = x, Y = y).    

    def _evaluate_p_pass(self, tau_arr, g_arr):
        """
        Evaluate the probability of passing the filter function.

        Arguments:
            F_sym: sympy expression
                The filter function $F(\beta)$.
            tau_arr: array(float)
                An array holding the values of $\tau_i$.
            g_arr: array(float)
                An array holding the values of $g_{\pm, i}$. g[i, 0] contains $g_{i,-}$ and g[i, 1] contains $g_{i,+}$.
        """
        F = self._substitute_guard_band_properties_into_F(self.F_sym, tau_arr, g_arr)

        integrand = sp.lambdify(
            (self.alpha_sym, self.beta_sym),
            sp.simplify((self.F_sym * self.Q_star_sym)),
            "numpy"
        )

        if self.NJIT:
            integrand = njit(integrand)

        # Integrate integrand over the entire complex plane.
        self.p_pass = nquad(integrand, [[-np.inf, np.inf], [-np.inf, np.inf]])

        return self.p_pass

    def _substitute_guard_band_properties_into_F(self, tau_arr, g_arr):
        """
        Substitute the guard band properties into the filter function $F(\beta)$.

                Arguments:
            F_sym: sympy expression
                The filter function $F(\beta)$.
            tau_arr: array(float)
                An array holding the values of $\tau_i$.
            g_arr: array(float)
                An array holding the values of $g_{\pm, i}$. g[i, 0] contains $g_{i,-}$ and g[i, 1] contains $g_{i,+}$.
        """
        return self.F_sym.subs(
            [(sp.Symbol(f'g_minus:{i}'), g_arr[i, 0]) for i in range(len(g_arr))] +
            [(sp.Symbol(f'g_plus:{i}'), g_arr[i, 1]) for i in range(len(g_arr))] +
            [(sp.Symbol(f'taus:{i}'), tau_arr[i]) for i in range(len(tau_arr))]
        )

    def _evaluate_marginals(self, x_range, y_range, num_points):
        """
            Wrapper function for _evaluate_marginals_uncompiled. This function is JIT compiled if the NJIT flag is set to True.
        """
        if self.NJIT:
            return njit(self._evaluate_marginals_uncompiled, parallel = True)(x_range, y_range, num_points)
        
        return self._evaluate_marginals_uncompiled(x_range, y_range, num_points)
    
    def _evaluate_marginals_uncompiled(self, x_range, y_range, num_points):
        """
        Given the form of Q_PS (in a numerically integrable lambda function), evaluate the marginal probability distributions p(X = x), p(Y = y) and the joint probability distribution p(X = x, Y = y).
        
        Arguments:
            x_range: array(float)
                The range of values for X.
            y_range: array(float)
                The range of values for Y.
            num_points: integer
                The number of points to evaluate the probability distributions at.
                For the joint probability distribution, this will be num_points^2, so be careful with large values.
        """
        if self.Q_PS_lambda is None:
            raise ValueError("The Q_PS lambda function has not been generated yet.")
        
        # Initialise arrays
        self.X = np.zeros(num_points)
        self.Y = np.zeros(num_points)
        self.XY = np.zeros((num_points, num_points))
        self.pX = np.zeros(num_points)
        self.pY = np.zeros(num_points)
        self.pXY = np.zeros((num_points, num_points))
        
        # Write x and y values into the first columns of pX and pY respectively.
        self.X = np.linspace(x_range[0], x_range[1], num_points)
        self.Y = np.linspace(y_range[0], y_range[1], num_points)

        # Distances between points. Useful for normalisation later.
        dx = self.X[1] - self.X[0]
        dy = self.Y[1] - self.Y[0]

        # Evaluate self.pXY via nquad. This is computationally expensive, but luckily pX and pY can be evaluated from pXY by using trapz across slices of pXY. This is true if the density of points is high enough.
        start_time = time.time()
        for i in prange(num_points):
            for j in prange(num_points):
                integrand = lambda alpha_re, beta_re : self.Q_PS_lambda(alpha_re, self.X[i], beta_re, self.Y[j])
                self.pXY[i][j] = nquad(integrand, [[-np.inf, np.inf], [-np.inf, np.inf]])[0]
        end_time = time.time()
        pXY_execution_time = end_time - start_time

        # Evaluate pX and pY is a similar manner (that is, using nquad).
        # There is a lot of repeated compute here, so consider switching to trapz if this is too slow.
        start_time = time.time()
        for i in prange(num_points):
            integrand = lambda alpha_im, beta_re, beta_im : self.Q_PS_lambda(self.X[i], alpha_im, beta_re, beta_im)
            self.pX[i] = nquad(integrand, [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])[0]
        end_time = time.time()
        pX_execution_time = end_time - start_time

        start_time = time.time()
        for i in prange(num_points):
            integrand = lambda alpha_re, alpha_im, beta_im : self.Q_PS_lambda(alpha_re, alpha_im, self.Y[i], beta_im)
            self.pY[i] = nquad(integrand, [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])[0]
        end_time = time.time()
        pY_execution_time = end_time - start_time

        return self.pXY, self.pX, self.pY, pXY_execution_time, pX_execution_time, pY_execution_time

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
            conditions.append(sp.And(g_minus[i] <= (sp.re(self.beta_sym) - taus[i]), (sp.re(self.beta_sym) - taus[i]) <= g_plus[i]))
            conditions.append(sp.And(g_minus[i] <= (sp.im(self.beta_sym) - taus[i]), (sp.im(self.beta_sym) - taus[i]) <= g_plus[i]))
        
        # Define the filter function. This unpacks conditions into essentially a huge OR statement.
        F = sp.Piecewise(
            (0, sp.Or(*conditions)),
            (1, True)
        )

        self.F_sym = F

        return self.F_sym

    def _generate_Q_star_lambda(self, a, b, c):
        """
        Generate the Q function (subject to no post-selection, hence Q_star) for a given covariance matrix in a lambda function, that is JIT compiled.
        IMPORTANT: Due to needing real integrands for scipy.integrate.nquad, the Q function is evaluated with the real and imaginary parts of alpha and beta as separate arguments.
        That is, Q_star_lambda(alpha_re, alpha_im, beta_re, beta_im).

        Arguments:
            a: float
                The coefficient $a$ of the covariance matrix.
            b: float
                The coefficient $b$ of the covariance matrix.
            c: float
                The coefficient $c$ of the covariance matrix.
        """
        self.Q_star_lambda = sp.lambdify(
            (self.alpha_re_sym, self.alpha_im_sym, self.beta_re_sym, self.beta_im_sym), # Function arguments
            self.Q_star_sym.subs(
                [
                    (self.alpha_sym, self.alpha_re_sym + self.alpha_im_sym*1j), # alpha = alpha_re + i alpha_im
                    (self.beta_sym, self.beta_re_sym + self.beta_im_sym*1j),    # beta = beta_re + i beta_im
                    (self.a_sym, a),
                    (self.b_sym, b),
                    (self.c_sym, c)]
            ).simplify(),
        )

        if self.NJIT:
            self.Q_star_lambda = njit(self.Q_star_lambda)

        return self.Q_star_lambda

    def _generate_Q_PS_lambda(self, a, b, c, tau_arr, g_arr):
        """
        Generate the Q function (subject to post-selection, hence Q_PS) for a given covariance matrix in a lambda function, that is JIT compiled.
        IMPORTANT: Due to needing real integrands for scipy.integrate.nquad, the Q function is evaluated with the real and imaginary parts of alpha and beta as separate arguments.
        That is, Q_PS_lambda(alpha_re, alpha_im, beta_re, beta_im).

        Arguments:
            a: float
                The coefficient $a$ of the covariance matrix.
            b: float
                The coefficient $b$ of the covariance matrix.
            c: float
                The coefficient $c$ of the covariance matrix.
            tau_arr: array(float)
                An array holding the values of $\tau_i$.
            g_arr: array(float)
                An array holding the values of $g_{\pm, i}$. g[i, 0] contains $g_{i,-}$ and g[i, 1] contains $g_{i,+}$.
        """
        
        # First, substitute the numerical guard band properties into the filter function.
        F = self._substitute_guard_band_properties_into_F(tau_arr, g_arr)

        # Find p_pass
        p_pass = self._evaluate_p_pass(tau_arr, g_arr)

        # Therefore, the composite Q function in symbolic form
        Q_PS = F * self.Q_star_sym / p_pass

        # Substitute and lambdify
        self.Q_PS_lambda = sp.lambdify(
            (self.alpha_re_sym, self.alpha_im_sym, self.beta_re_sym, self.beta_im_sym),                                                                      
            Q_PS.subs(
                [
                    (self.alpha_sym, self.alpha_re_sym + self.alpha_im_sym*1j), # alpha = alpha_re + i alpha_im
                    (self.beta_sym, self.beta_re_sym + self.beta_im_sym*1j),    # beta = beta_re + i beta_im
                    (self.a_sym, a),
                    (self.b_sym, b),
                    (self.c_sym, c)]
            ).simplify(),
            "numpy"
        )

        if self.NJIT:
            self.Q_PS_lambda = njit(self.Q_PS_lambda)

        return self.Q_PS_lambda

class GBSR(GBSR_quantum_statistics):
    def __init__(self, m) -> None:
        super().__init__()

        #self.F_sym = self._define_filter_function(m)
        #self.F_sym_subs = self._substitute_guard_band_properties_into_F(tau_arr, g_arr)

        #self.Q_PS_lambda = self._generate_Q_PS_lambda(a, b, c, tau_arr, g_arr)
        
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
    
