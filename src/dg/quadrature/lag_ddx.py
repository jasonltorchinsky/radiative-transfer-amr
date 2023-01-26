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
            dmat[j, i] = lag_deriv(nodes, i, nodes[j])

    return dmat

def lag_deriv(nodes, i, x):
    '''
    Calculates the derivative of the ith Lagrange polynomial for a given set of
    nodes.
    '''

    nnodes = np.shape(nodes)[0]

    dbase = 0
    for m in range(0, nnodes):
        if m != i:
            prod = 1
            for j in range(0, nnodes):
                if (j != i) and (j != m):
                    prod *= (x - nodes[j]) / (nodes[i] - nodes[j])

            prod /= (nodes[i] - nodes[m])
            dbase += prod

    return dbase
