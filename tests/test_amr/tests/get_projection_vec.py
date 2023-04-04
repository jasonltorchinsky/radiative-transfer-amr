import numpy as np
import sys

sys.path.append('../../src')
from dg.matrix import get_idx_map, get_col_idxs, get_cell_idxs
from dg.projection import Projection, push_forward
import dg.quadrature as qd

def get_projection_vec(mesh, f):
    """
    Create the global vector corresponding to the projection of some
    function f.
    """

    f_proj = Projection(mesh, f)
    f_vec = f_proj.to_vector()
    
    return f_vec
