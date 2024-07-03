import numpy as np
import sympy as sp
from numba import njit
from scipy.stats import norm

class GBSR_quantum_statistics():
    def __init__(self, modulation_variance, transmittance, excess_noise) -> None:
        """
        Class arguments:
            modulation_variance: Alice's modulation variance $V_\\text{mod}$.
            transmittance: Transmissivity $T$ of the channel.
            excess_noise: Excess noise $\\xi$ of the channel.
        """

        self.modulation_variance = modulation_variance
        self.transmittance = transmittance
        self.excess_noise = excess_noise

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

        # Define the Husimi-Q function of a TMSV vacuum state, subject to no post-selection, in sympy.
        self.Q_star = (sp.sqrt(self.cov_mat.det()) / sp.pi**2) * sp.exp(-self.a_sym * (sp.Abs(self.alpha_sym))**2 - self.b_sym * sp.Abs(self.beta_sym)**2 - 2 * self.c_sym * sp.Abs(self.alpha_sym) * sp.Abs(self.beta_sym) * sp.cos(sp.arg(self.alpha_sym) - sp.arg(self.beta_sym)))

        # Generate the Q_star lambda function with, a, b and c as parameters. This must be updated when a, b and c change.
        # This should probably be done as a class property but it's likely just me using this code so this is probably okay.
        self.Q_star_lambda = self.generate_Q_star_lambda(
            self.alice_variance,
            self.bob_variance,
            np.sqrt(self.transmittance * (self.alice_variance*self.alice_variance - 1))
        )

    def generate_Q_star_lambda(self, a, b, c):
        """
        Generate the Q function for a given covariance matrix in a lambda function, that is JIT compiled.
        """
        return njit(
            sp.lambdify(
                (self.alpha_sym, self.beta_sym), # Function arguments
                self.Q_star.subs([(self.a_sym, a), (self.b_sym, b), (self.c_sym, c)]),
            )
        )

class GBSR(GBSR_quantum_statistics):
    def __init__(self) -> None:
        super().__init__()

    def evaluate_key_rate_in_bits_per_pulse(self, m: int, interval_edges: list, gb_widths: list[list, list]):
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
        
        quantisation_entropy = integrator.evaluate_slicing_entropy(m, interval_edges)
        leaked_classical_information = integrator.evaluate_error_correction_information(m, interval_edges)

        return ((quantisation_entropy - leaked_classical_information) / self.mutual_information) * (self.mutual_information - self.holevo_information)

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
    
