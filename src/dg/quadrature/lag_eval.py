import numpy as np

def lag_eval(nodes, i, x):
    '''
    Evaluate the ith Lagrange polynomial for a given set of nodes.
    '''

    # Only evaluate non-zero for pull-back coordinates
    if ((-1.0 <= x) and (x <= 1.0)):
        order = np.shape(nodes)[0] - 1
        res = 1
        for j in range(0, order + 1):
            if j != i:
                res *= (x - nodes[j]) / (nodes[i] - nodes[j])
    else:
        return 0

    return res