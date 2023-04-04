import numpy as np
from scipy.sparse.linalg import spsolve
import sys

from .calc_mass_matrix import calc_mass_matrix
from .calc_scat_matrix import calc_scat_matrix
from .calc_intr_conv_matrix import calc_intr_conv_matrix
from .calc_bdry_conv_matrix import calc_bdry_conv_matrix
from .calc_forcing_vec import calc_forcing_vec
from .calc_bcs_vec import calc_bcs_vec

from utils import print_msg

sys.path.append('../src')
from dg.matrix import get_intr_mask, split_matrix, merge_vectors
from dg.projection import push_forward, to_projection

def rtdg(mesh, kappa, sigma, Phi, bcs, f = None):
    """
    Solve the RT problem.
    """
    
    M_mass      = calc_mass_matrix(mesh, kappa)
    M_scat      = calc_scat_matrix(mesh, sigma, Phi)
    M_intr_conv = calc_intr_conv_matrix(mesh)
    M_bdry_conv = calc_bdry_conv_matrix(mesh)
    
    M = (M_bdry_conv - M_intr_conv) + M_mass - M_scat
    intr_mask = get_intr_mask(mesh)
    [M_intr, M_bdry] = split_matrix(mesh, M, intr_mask)
    
    if f is None:
        def forcing(x, y, th):
            return 0
    else:
        def forcing(x, y, th):
            return f(x, y)
    
    bcs_vec = calc_bcs_vec(mesh, bcs)
    
    bdry_mask = np.invert(intr_mask)
    
    f_vec = calc_forcing_vec(mesh, forcing)
    f_intr_vec = f_vec[intr_mask]
    
    u_intr_vec = spsolve(M_intr, f_intr_vec - M_bdry @ bcs_vec)
    u_vec  = merge_vectors(u_intr_vec, bcs_vec, intr_mask)
    u_proj = to_projection(mesh, u_vec)

    return u_proj

def merge_vecs(intr_mask, intr_vec, bdry_vec):
    
    ndof = np.size(intr_mask)

    vec = np.zeros(ndof)
    intr_idx = 0
    bdry_idx = 0

    for ii in range(0, ndof):
        if intr_mask[ii]:
            vec[ii] = intr_vec[intr_idx]

            intr_idx += 1
        else:
            vec[ii] = bdry_vec[bdry_idx]
            
            bdry_idx += 1

    return vec
