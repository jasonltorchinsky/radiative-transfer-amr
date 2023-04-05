import numpy as np

from .calc_proj_mtx_1d import calc_proj_mtx_1d

def proj_1d(func, src_nodes, trgt_nodes):
    '''
    func - Analytic function to project.
    src_nodes - Input data coordinates.
    trgt_nodes - Output data locations.

    Returns the projection of an analytic function from one nodal
    basis (src_nodes) onto another one (trgt_nodes).
    '''

    f_src = func(src_nodes)

    P = calc_proj_mtx_1d(src_nodes, trgt_nodes) # Projection matrix

    return P @ f_src
