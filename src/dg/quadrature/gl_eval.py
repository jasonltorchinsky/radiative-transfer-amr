import numpy as np

def gl_eval(nodes, i, x):
    '''
    Evaluate the ith Gauss-Lobatto basis function (i.e., Legendre polynomial
    for the nodes) at x.
    '''

    order = np.shape(nodes)[0] - 1
    res = 1
    for j in range(0, order + 1):
        if j != i:
            res *= (x - nodes[j]) / (nodes[i] - nodes[j])

    return res
