# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports

INT: type  = np.int64
REAL: type = np.float64

PI: float  = np.pi
EPS: float = np.finfo(REAL).resolution
INF: float = np.finfo(REAL).max