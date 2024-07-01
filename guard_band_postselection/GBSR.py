import numpy as np
from scipy.stats import norm

# TODO: Fill out use of mrmustard for rigorous introduction of statistics. That is, proper marginals calculated from the proper Q function.
# For now, we'll just use norm from scipy.stats.

class quantum_statistics():
    def __init__(self, V_mod, transmittance, excess_noise):
        """
        Class arguments:
            V_mod: Alice's modulation variance $V_\\text{mod}$.
            trans: Transmissivity $T$ of the channel.
            excess_noise: Excess noise $\\xi$ of the channel.
        """

        # TODO: Make variance parameters here immutable by making them properties of the class. Hopefully underscore prefixes are enough for now.

        # With channel parameters passed, we can digest other relevant parameters.
        self.V_mod = V_mod
        self.T = transmittance
        self.xi = excess_noise

        self.V_A = V_mod + 1.0                             # Alice's effective variance $V_A = $V_\\text{mod} + 1$ in SNU.
        self.V_B = (self.T * self.V_mod) + 1 + self.xi     # Bob's effective variance $V_B = T V_\\text{mod} + 1 + \\xi$ in SNU.

        # Covariance matrix parameters. See first year report or (Laudenbach-2018) for details.
        # In short, $a$ is Alice's variance, $b$ is Bob's variance and $c$ is the cross-correlation term.
        self._a = self.V_A
        self._b = self.V_B
        self._c = np.sqrt(self.T * (self.V_A*self.V_A - 1))

        # We can then use these to build the 4x4 covariance matrix:
        #
        #           
        #               a\mathbb{I}_2  c\sigma_z
        # cov_mat =  (                            )
        #                c\sigma_z  b\mathbb{I}_2
        #
        #
        self.cov_mat = np.array(
            [
                [self._a,       0,          self._c,         0],
                [0,             self._a,    0,        -self._c],
                [self._c,       0,          self._b,         0],
                [0,             -self._c,   0,         self._b]
            ]
        )
        self._cov_mat_det = np.linalg.det(self.cov_mat)
        self._sqrt_cov_mat_det = np.sqrt(self._cov_mat_det)

        # Initialise instances of scipy.stats.norm objects for Alice and Bob.
        # Stats can then be gathered via self.alice_norm.pdf, for example.
        self.alice_norm = norm(loc = 0.0, scale = np.sqrt(self.V_A))
        self.bob_norm   = norm(loc = 0.0, scale = np.sqrt(self.V_B))


class sec_no_post_selection_key_rate(quantum_statistics):
    def __init__(self, v_mod, transmittance, excess_noise):
        """
            Run quantum_statistics.__init__() to initialise appropriate parameters.
        """
        self.super().__init__(v_mod, transmittance, excess_noise)

        self.mutual_information = self._evaluate_mutual_information()
        self.holevo_information = self._evaluate_holevo_information()

    def evaluate(self, m, interval_edges):
        """
            Evaluate the key rate for the no post-selection case (for now) with a given number of slices $m$ and an array holding each interval edge.

            Arguments:
                m: integer
                    The number of slices.
                interval_edges: array(flaot)
                    An array holding the edges of each interval, from left to right. 
                    That is, the first interval is -np.inf and the last is np.inf.
                    Of course, the non-extremal values of the array should be finite and in ascending order.
                    The interval edges should be given in units of standard deviation. That is, an interval which is one standard deviation from the mean would have a value of 1.0.
                    
            Returns:
                key_rate: float
                    The key rate for the given number of slices and interval edge positions.
        """
        integrator = sliced_error_correction_integrator(m, interval_edges)
        
        quantisation_entropy = integrator.evaluate_slicing_entropy(m, interval_edges)
        error_correction_information = integrator.evaluate_error_correction_information(m, interval_edges)

        return ((quantisation_entropy - error_correction_information) / self.mutual_information) * (self.mutual_information - self.holevo_information)

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
    
