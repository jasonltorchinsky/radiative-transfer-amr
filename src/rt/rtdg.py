import numpy as np
from scipy.sparse.linalg import bicg, bicgstab, cg, cgs, gmres, lgmres, \
                                minres, qmr, gcrotmk, tfqmr, spsolve, inv
import sys
from time import perf_counter

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

def rtdg(mesh, kappa, sigma, Phi, bcs_dirac, f = None, **kwargs):
    """
    Solve the RT problem.
    """
    
    default_kwargs = {'solver' : 'spsolve',
                      'precondition' : False,
                      'verbose' : False}
    kwargs = {**default_kwargs, **kwargs}
    
    M_mass      = calc_mass_matrix(mesh, kappa, **kwargs)
    M_scat      = calc_scat_matrix(mesh, sigma, Phi, **kwargs)
    M_intr_conv = calc_intr_conv_matrix(mesh, **kwargs)
    M_bdry_conv = calc_bdry_conv_matrix(mesh, **kwargs)
    
    M_conv = M_bdry_conv - M_intr_conv
    M = M_conv + M_mass - M_scat
    intr_mask = get_intr_mask(mesh)
    [M_intr, M_bdry] = split_matrix(mesh, M, intr_mask)
    
    if f is None:
        def forcing(x, y, th):
            return 0
    else:
        def forcing(x, y, th):
            return f(x, y)
    
    bcs_vec = calc_bcs_vec(mesh, bcs_dirac)
    
    bdry_mask = np.invert(intr_mask)
    
    f_vec = calc_forcing_vec(mesh, forcing)
    f_intr_vec = f_vec[intr_mask]
    
    A = M_intr
    b = f_intr_vec - M_bdry @ bcs_vec
    
    
    if kwargs['verbose']:
        t0 = perf_counter()
        
    # Calculate preconditioner
    if kwargs['precondition']:
        M_pc = inv(M_conv + M_mass)
        [M_pc, _] = split_matrix(mesh, M_pc, intr_mask)
    else:
        M = None
    
    if kwargs['solver'] == 'bicg':
        u_intr_vec = bicg(A, b, M = M_pc)
    elif kwargs['solver'] == 'bicgstab':
        u_intr_vec = bicgstab(A, b, M = M_pc)
    elif kwargs['solver'] == 'cg':
        u_intr_vec = cg(A, b, M = M_pc)
    elif kwargs['solver'] == 'cgs':
        u_intr_vec = cgs(A, b, M = M_pc)
    elif kwargs['solver'] == 'gmres':
        u_intr_vec = gmres(A, b, M = M_pc)
    elif kwargs['solver'] == 'lgmres':
        u_intr_vec = lgmres(A, b, M = M_pc)
    elif kwargs['solver'] == 'minres':
        u_intr_vec = minres(A, b, M = M_pc)
    elif kwargs['solver'] == 'qmr':
        u_intr_vec = qmr(A, b, M1 = M_pc)
    elif kwargs['solver'] == 'gcrotmk':
        u_intr_vec = gcrotmk(A, b, M = M_pc)
    elif kwargs['solver'] == 'tfqmr':
        u_intr_vec = tfqmr(A, b, M = M_pc)
    else:
        u_intr_vec = spsolve(A, b)
        
    if kwargs['verbose']:
        tf = perf_counter()
        if kwargs['precondition']:
            prec_str = 'Preconditioned-'
        else:
            prec_str = ''
        msg = (
            'Solver {}{}:\n'.format(prec_str, kwargs['solver']) +
            'Solve Time {:8.4f}\n'.format(tf - t0)
            )
        print_msg(msg)
    
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
