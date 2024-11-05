# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports

# Relative Imports

def push_forward(x0: float, xf: float, nodes: np.ndarray) -> np.ndarray:
    """
    Transforms nodes on [-1, 1] to a [x0, xf].
    """

    xx: np.ndarray = x0 + (xf - x0) / 2.0 * (nodes + 1.0)

    return xx

def pull_back(x0: float, xf: float, nodes: np.ndarray) -> np.ndarray:
    """
    Transforms nodes on [x0, xf] to a [-1, 1].
    """

    xx: np.ndarray = -1.0 +  2.0 / (xf - x0) * (nodes - x0)

    return xx
