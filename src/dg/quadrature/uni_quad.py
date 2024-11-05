import numpy as np

def uni_quad(nnodes):
    """
    Computes the nodes and weights needed for integration via a uniform
    quadrature.

    This is NOT meant for actual use, only for comparison against other
    quadrature rules.
    """
    
    nodes = np.linspace(-1, 1, nnodes)
    weights = 2. * np.ones(nnodes)
    
    return [nodes, weights]
