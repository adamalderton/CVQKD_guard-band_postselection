import numpy as np

from .integration import sliced_error_correction_integrator
from .quantum_statistics import quantum_statistics

# TODO: Uses integrator classes to abstract integration away from here. This file and classes within it should be used to bring together terms.
# TODO: Use quantum_statistics class to read out mutual informations.

class sec_key_rate():
    def __init__(self):
        pass