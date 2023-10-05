import numpy as np
from scipy.stats import norm

# TODO: Fill out use of mrmustard for rigorous introduction of statistics. That is, proper marginals calculated from the proper Q function.
# For now, we'll just use norm from scipy.stats.

# TODO: Initialise as a covariance matrix following obsidian notes.
# TODO: Return an instance of a scipy.stats.norm object to integrate with.

class quantum_statistics():
    def __init__(self, v_mod, transmittance, excess_noise):
        """
        Class arguments:
            v_mod: Alice's modulation variance $V_\\text{mod}$.
            trans: Transmissivity $T$ of the channel.
            excess_noise: Excess noise $\\xi$ of the channel.
        """

        # With channel parameters passed, we can digest other relevant parameters.
        self.v_mod = v_mod
        self.T = transmittance
        self.xi = excess_noise

        self._v_A = v_mod + 1.0                             # Alice's effective variance $V_A = $V_\\text{mod} + 1$ in SNU.
        self._v_B = (self.T * self.v_mod) + 1 + self.xi     # Bob's effective variance $V_B = T V_\\text{mod} + 1 + \\xi$ in SNU.

        # Covariance matrix parameters. See first year report or (Laudenbach-2018) for details.
        # In short, $a$ is Alice's variance, $b$ is Bob's variance and $c$ is the cross-correlation term.
        self._a = self._v_A
        self._b = self._v_B
        self._c = np.sqrt(self.T * (self._v_A*self._v_A - 1))

        # We can then use these to build the 4x4 covariance matrix:
        #
        #           
        #           a\mathbb{I}_2 & c\sigma_z
        #  cov =  (                            )
        #           c\sigma_z & b\mathbb{I}_2
        #
        #
        self.cov = np.array(
            [
                [self._a,       0,          self._c,         0],
                [0,             self._a,    0,        -self._c],
                [self._c,       0,          self._b,         0],
                [0,             -self._c,   0,         self._b]
            ]
        )

        