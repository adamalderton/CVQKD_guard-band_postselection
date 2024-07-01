import numpy as np
from numba import njit
from scipy.integrate import dblquad

from .quantum_statistics import quantum_statistics

# TODO: DO ONE FOR MSB AND ONE FOR LSB

class sliced_error_correction_integrator():
    def __init__(self, m, interval_edges, alice_variance, bob_variance, cross_correlation):
        """
            Class to perform necessary integrations for evaluate informations as part of the key rate.

            Arguments:
                m: integer
                    The number of slices.
                interval_edges: array(flaot)
                    An array holding the edges of each interval, from left to right. 
                    That is, the first interval is -np.inf and the last is np.inf.
                    Of course, the non-extremal values of the array should be finite and in ascending order.
                    The interval edges should be given in units of standard deviation. That is, an interval which is one standard deviation from the mean would have a value of 1.0.
        """

        self.m = m
        self.interval_edges = interval_edges

        self.sigma_X = sigma_X
        self.sigma_noise = sigma_noise

        self.var_X = sigma_X * sigma_X
        self.var_noise = sigma_noise * sigma_noise

        self.
    
    def evaluate_slicing_entropy(self):
        """
            Evaluate the quantisation (slicing) entropy for the given number of slices and interval edge positions.
        """
    # TODO: Change this! Going to have to integrate over renormalised pdf
        probs = [self._integrate_gaussian_pdf((self.c[j], self.c[j + 1]), self.sigma_X) for j in range(0, 4)]
        return -1.0 * np.sum([p * np.log2(p) for p in probs])

    def evaluate_error_correction_information(self):
        """
            Evaluate the error correction information for the given number of slices and interval edge positions.
        """
        pass
    
    @njit
    def _joint_pdf(self, Y, X):
        """
            The function to be numerically integrated by an external function.
            That is, this is a double integration that should be invoked by scipy.integrate.dblquad.
        """
        YmX = Y - X
        
        return \
            ( 1.0 / (2*np.pi * self.sigma_X * self.sigma_noise) ) * \
            np.exp(- 0.5 * (
                    ( (X*X) / (self.var_X) ) + \
                    ( (YmX*YmX) / (self.var_noise) )
                )
            )

    @njit
    def _integrate_joint_pdf(self, xlims, ylims):
        # Integrate the joint PDF over the given limits.

        return dblquad(
            self._joint_pdf,
            xlims[0], # X lower limit
            xlims[1], # X upper limit
            ylims[0], # Y lower limit
            ylims[1], # Y upper limit
        )[0]

    @njit
    def _interval_overlap_no_postselection(self, j, k):
        """
            Evaluates the overlap integral of x originally in interval j being (erroneously) measured as y in interval k in the case of no post-selection.
        """
        return self._integrate_joint_pdf(
            (self.c[j], self.c[j + 1]),                 # X limits
            (self.b_pos[k][1], self.b_pos[k + 1][0])    # Y limits (range of interval k that is NOT inside any guard bands).
        )

    def _binary_entropy(self, e):
        me = 1.0 - e
        return -1.0 * (e * np.log2(e) + me * np.log2(me))
