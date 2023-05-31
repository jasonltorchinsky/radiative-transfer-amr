import numpy as np

from .lag_eval import lag_eval

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

def lag_ddx_eval_approx(nodes, j, x):
    '''
    Calculates the *approximate value* of the derivative of the jth Lagrange
    polynomial for a given set of nodes.
    '''

    nnodes = np.shape(nodes)[0]
    
    xm = x - 10.**(-10)
    if xm < -1:
        xm = -1.
    
    xp = x + 10.**(-10)
    if xp > 1:
        xp = 1
    
    dx = xp - xm
    
    fm = lag_eval(nodes, j, xm)
    fp = lag_eval(nodes, j, xp)
    res = (fp - fm) / dx
    
    return res

def lag_ddx_eval(nodes, j, x):
    '''
    Calculates the exact value of the derivative of the jth Lagrange polynomial
    for a given set of  nodes.
    '''

    nnodes = np.shape(nodes)[0]

    res = 0
    for kk in range(0, nnodes):
        if kk != j:
            prod = 1
            for ii in range(0, nnodes):
                if (ii != j) and (ii != kk):
                    prod *= (x - nodes[ii]) / (nodes[j] - nodes[ii])

            res += prod / (nodes[j] - nodes[kk])

    return res
