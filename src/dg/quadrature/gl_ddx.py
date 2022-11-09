import numpy as np

def gl_ddx(nodes):
    '''
    Returns differentiation matrix for a function f evaluated at the nodes.
    E.g., sum_j dmat_{ij} * f(pt(j)) appriximates d_x f(pt(i)) 
    '''

    order = np.shape(nodes)[0] - 1
    dmat = np.zeros([order + 1, order + 1])

    for j in range(0, order + 1):
        for i in range(0, order + 1):
            dmat[j, i] = gl_deriv(nodes, i, nodes[j])

    return dmat

def gl_deriv(nodes, i, x):
    '''
    Calculates the derivative of the ith Gauss-Lobatto basis function at x.
    '''

    order = np.shape(nodes)[0] - 1

    dbase = 0
    for m in range(0, order + 1):
        if m != i:
            prod = 1
            for j in range(0, order + 1):
                if (j != i) and (j != m):
                    prod *= (x - nodes[j]) / (nodes[i] - nodes[j])

            prod /= (nodes[i] - nodes[m])
            dbase += prod

    return dbase
