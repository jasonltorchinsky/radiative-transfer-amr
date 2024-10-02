import numpy as np
from numpy.linalg import inv

from .gl_eval import gl_eval

def gl_proj_2D(xx, nodes_x, yy, nodes_y, comp_mpinv):
    """
    xx, yy  - Output data locations
    nodes_x, nodes_y - input data locations

    Returns the coefficient matrix for projecting a function onto the 
    Gauss-Lobatto basis (e.g., Legendre Polynomials at the nodes.
    Specifically, for f = sum_i k_i * phi_i for basis functions i, 
    we ideally have A k = d, where d if the vector of f evaluated at the
    xxs.

    However, is A is not square, we instead solve A^T A k = A^T d,
    which leads to k = (A^T A)^-1 A^T d.

    The final flag is whther or not to calculate the Moore-Penrose inverse.
    """

    nx = np.shape(xx)[0]
    ny = np.shape(yy)[0]
    order_x = np.shape(nodes_x)[0] - 1
    order_y = np.shape(nodes_y)[0] - 1
    
    A = np.zeros([nx*ny, (order_x + 1) * (order_y + 1)])
    for i in range(0, nx*ny):
        x_idx = int(np.mod(i, nx))
        y_idx = int(np.floor(i / nx))
        for j in range(0, (order_x + 1) * (order_y + 1)):
            ox_idx = int(np.mod(j, (order_x + 1)))
            oy_idx = int(np.floor(j / (order_x + 1)))
            A[i, j] = gl_eval(nodes_x, ox_idx, xx[x_idx]) \
                * gl_eval(nodes_y, oy_idx, yy[y_idx])

    if comp_mpinv:
        Adag = inv(A.transpose() @ A) @ A.transpose()
    else:
        Adag = None

    return [A, Adag]
