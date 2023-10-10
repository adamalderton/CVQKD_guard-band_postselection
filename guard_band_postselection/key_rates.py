import numpy as np

from .integration import sliced_error_correction_integrator
from .quantum_statistics import quantum_statistics

# TODO: Uses integrator classes to abstract integration away from here. This file and classes within it should be used to bring together terms.
# TODO: Use quantum_statistics class to read out mutual informations.

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
    
if __name__ == "__main__":
    v_mod = 2.2
    T = 0.4
    xi = 0.05

    m = 1
    interval_edges = np.array([-np.inf, 0.0, np.inf])

    print(sec_no_post_selection_key_rate(v_mod, T, xi).evaluate(m, interval_edges))

    