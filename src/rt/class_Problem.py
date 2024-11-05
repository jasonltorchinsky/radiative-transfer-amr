# Standard Library Imports
from typing import Callable

# Third-Party Library Imports
from numpy import ndarray as array

# Local Library Imports

# Relative Imports

class Problem():
    def __init__(self,
                 kappa: Callable[[array, array, array], array],
                 sigma: Callable[[array, array, array], array],
                 Phi: Callable[[array, array], array],
                 f: Callable[[array, array, array], array],
                 bcs_dirac: list):
        
        self.kappa: Callable[[array, array, array], array] = kappa
        self.sigma: Callable[[array, array, array], array] = sigma
        self.Phi: Callable[[array, array], array] = Phi
        self.f: Callable[[array, array, array], array] = f
        self.bcs_dirac: list = bcs_dirac

    from .boundary_conditions_vector import boundary_conditions_vector
    from .boundary_convection_matrix import boundary_convection_matrix
    from .forcing_vector import forcing_vector
    from .interior_convection_matrix import interior_convection_matrix
    from .mass_matrix import mass_matrix
    from .preconditioner_matrix import preconditioner_matrix
    from .scattering_matrix import scattering_matrix
    from .solve import solve
