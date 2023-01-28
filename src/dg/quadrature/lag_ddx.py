import numpy as np

def lag_ddx(nodes):
    '''
    Returns differentiation matrix for a function f evaluated at the nodes.
    E.g., sum_j dmat_{ij} * f(pt(j)) appriximates d_x f(pt(i)) 
    '''

    nnodes = np.shape(nodes)[0]
    dmat   = np.zeros([nnodes, nnodes])

    for j in range(0, nnodes):
        for i in range(0, nnodes):
            dmat[j, i] = lag_ddx_eval(nodes, i, nodes[j])

    return dmat

def lag_ddx_eval(nodes, i, x):
    '''
    Calculates the derivative of the ith Lagrange polynomial for a given set of
    nodes.
    '''

    nnodes = np.shape(nodes)[0]

    res = 0
    for mm in range(0, nnodes):
        if mm != i:
            prod = 1
            for nn in range(0, nnodes):
                if (nn != i) and (nn != mm):
                    prod *= (x - nodes[nn]) / (nodes[i] - nodes[nn])

            prod /= (nodes[i] - nodes[mm])
            res += prod

    return res
