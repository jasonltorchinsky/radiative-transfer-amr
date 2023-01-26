import numpy as np
from numpy.linalg import inv

from .lag_eval import lag_eval

def calc_proj_mtx_1d(src_nodes, trgt_nodes):
    '''
    src_nodes - Input data coordinates.
    trgt_nodes - Output data locations.

    Returns the coefficient matrix for projecting a function from one nodal
    basis (src_nodes) onto another one (trgt_nodes).
    '''

    n_src = np.shape(src_nodes)[0]
    n_trgt = np.shape(trgt_nodes)[0]

    # Have more source information, use source quadrature rule to get
    # projection matrix (P) form. Results in equation like src = P * trgt,
    # so return Moore-Penrose inverse of P.
    if (n_src > n_trgt):
        P = np.zeros([n_src, n_trgt])
        for ii in range(0, n_src):
            for jj in range(0, n_trgt):
                P[ii, jj] = lag_eval(trgt_nodes, jj, src_nodes[ii])

        return inv(P.transpose() @ P) @ P.transpose()

    elif(n_src <= n_trgt):
        P = np.zeros([n_trgt, n_src])
        for jj in range(0, n_trgt):
            for ii in range(0, n_src):
                P[jj, ii] = lag_eval(src_nodes, ii, trgt_nodes[jj])

        return P

    else:
        return None
