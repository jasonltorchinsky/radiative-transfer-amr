import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag, bmat

from .Projection import Projection_2D

import dg.quadrature as qd
from dg.mesh import ji_mesh, tools

def push_forward(x0, xf, nodes):
    """
    Transforms nodes on [-1, 1] to a [x0, xf].
    """

    xx = x0 + (xf - x0) / 2 * (nodes + 1)

    return xx

def pull_back(x0, xf, nodes):
    """
    Transforms nodes on [x0, xf] to a [-1, 1].
    """

    xx = -1 +  2 / (xf - x0) * (nodes - x0)

    return xx

def get_col_info(col):
    """
    Get column dimensions, number of DOFs, quadrature weights, etc.
    """

    [x0, y0, x1, y1] = col.pos
    dx = x1 - x0
    dy = y1 - y0
    [dof_x, dof_y] = col.ndofs
    
    [_, weights_x, _, weights_y, _, _] = qd.quad_xya(dof_x, dof_y, 1)

    return [x0, y0, x1, y1, dx, dy, dof_x, dof_y, weights_x, weights_y]

def get_cell_info(cell):
    """
    Get cell dimension, number of DOFs, quadrature weights, etc.
    """
    
    [a0, a1] = cell.pos
    da = a1 - a0
    [dof_a] = cell.dofs
    [_, _, _, _, nodes_a, weights_a] = qd.quad_xya(1, 1, dof_a)

    return [a0, a1, da, dof_a, nodes_a, weights_a]
