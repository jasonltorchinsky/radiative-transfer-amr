import numpy as np
from numpy.linalg import inv

from .gl_eval import gl_eval

def gl_proj(xx, nodes, comp_mpinv):
    '''
    xx  - Output data locations
    nodes - input data locations

    Returns the coefficient matrix for projecting a function onto the 
    Gauss-Lobatto basis (e.g., Legendre Polynomials at the nodes.
    Specifically, for f = sum_i k_i * phi_i for basis functions i, 
    we ideally have A k = d, where d if the vector of f evaluated at the
    xxs.

    However, is A is not square, we instead solve A^T A k = A^T d,
    which leads to k = (A^T A)^-1 A^T d.

    The final flag is whther or not to calculate the Moore-Penrose inverse.
    '''

    nx = np.shape(xx)[0]
    order = np.shape(nodes)[0] - 1
    
    A = np.zeros([nx, order + 1])
    for i in range(0, nx):
        for j in range(0, order + 1):
            A[i, j] = gl_eval(nodes, j, xx[i])

    if comp_mpinv:
        Adag = inv(A.transpose() @ A) @ A.transpose()
    else:
        Adag = None

    return [A, Adag]
