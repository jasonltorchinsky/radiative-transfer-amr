import numpy as np
from inspect import signature
from scipy.sparse.linalg import bicg, bicgstab, cg, cgs, gmres, lgmres, \
                                minres, qmr, gcrotmk, tfqmr, spsolve, inv
import sys
from time import perf_counter

from .calc_mass_matrix import calc_mass_matrix
from .calc_scat_matrix import calc_scat_matrix
from .calc_precond_matrix import calc_precond_matrix
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
    
    default_kwargs = {'solver'       : 'spsolve',
                      'precondition' : False,
                      'verbose'      : False,
                      'x0'           : None, # Includes boundary data
                      'tol'          : 1.e-8
                      } 
    kwargs = {**default_kwargs, **kwargs}

    if kwargs['verbose']:
        tf = perf_counter()
        if kwargs['precondition']:
            prec_str = 'Preconditioned-'
        else:
            prec_str = ''
        msg = (
            'DoFs: {}, Solver {}{}.\n'.format(mesh.get_ndof(), prec_str, kwargs['solver'])
            )
        print_msg(msg)
    
    [M_mass_scat, M_pc] = calc_precond_matrix(mesh, kappa, sigma, Phi, **kwargs)
    M_intr_conv = calc_intr_conv_matrix(mesh, **kwargs)
    M_bdry_conv = calc_bdry_conv_matrix(mesh, **kwargs)
    
    M_conv = M_bdry_conv - M_intr_conv
    M = M_conv + M_mass_scat
    intr_mask = get_intr_mask(mesh)
    [M_intr,    M_bdry] = split_matrix(mesh, M, intr_mask)
    if kwargs['precondition']:
        [M_pc, _]       = split_matrix(mesh, M_pc, intr_mask)
    else:
        M_pc = None

    # Make sure forcing function takes three arguments
    if f is None:
        def forcing(x, y, th):
            return 0
    if len(signature(f).parameters) == 1:
        def forcing(x, y, th):
            return f(x)
    elif len(signature(f).parameters) == 2:
        def forcing(x, y, th):
            return f(x, y)
    elif len(signature(f).parameters) == 3:
        def forcing(x, y, th):
            return f(x, y, th)
    
    bcs_vec = calc_bcs_vec(mesh, bcs_dirac)
    
    bdry_mask = np.invert(intr_mask)
    
    f_vec = calc_forcing_vec(mesh, forcing)
    f_intr_vec = f_vec[intr_mask]
    
    A = M_intr
    b = f_intr_vec - M_bdry @ bcs_vec
    
    if kwargs['verbose']:
        t0 = perf_counter()

    tol = kwargs['tol']
    if kwargs['x0'] is not None:
        x0 = kwargs['x0']
        x0 = x0[intr_mask]
    else:
        x0 = None
    if kwargs['solver'] == 'bicg':
        [u_intr_vec, info] = bicg(    A, b, x0 = x0, tol = tol, M = M_pc)
    elif kwargs['solver'] == 'bicgstab':
        [u_intr_vec, info] = bicgstab(A, b, x0 = x0, tol = tol, M = M_pc)
    elif kwargs['solver'] == 'cg':
        [u_intr_vec, info] = cg(      A, b, x0 = x0, tol = tol, M = M_pc)
    elif kwargs['solver'] == 'cgs':
        [u_intr_vec, info] = cgs(     A, b, x0 = x0, tol = tol, M = M_pc)
    elif kwargs['solver'] == 'gmres':
        [u_intr_vec, info] = gmres(   A, b, x0 = x0, tol = tol, M = M_pc)
    elif kwargs['solver'] == 'lgmres':
        [u_intr_vec, info] = lgmres(  A, b, x0 = x0, tol = tol, M = M_pc)
    elif kwargs['solver'] == 'qmr':
        [u_intr_vec, info] = qmr(     A, b, x0 = x0, tol = tol, M1 = M_pc)
    elif kwargs['solver'] == 'gcrotmk':
        [u_intr_vec, info] = gcrotmk( A, b, x0 = x0, tol = tol, M = M_pc)
    elif kwargs['solver'] == 'tfqmr':
        [u_intr_vec, info] = tfqmr(   A, b, x0 = x0, tol = tol, M = M_pc)
    elif kwargs['solver'] == 'spsolve':
        u_intr_vec = spsolve(A, b)
        info = 0
    else:
        kwargs['solver'] = 'spsolve'
        u_intr_vec = spsolve(A, b)
        info = 0
        
    if kwargs['verbose']:
        tf = perf_counter()
        if kwargs['precondition']:
            prec_str = 'Preconditioned-'
        else:
            prec_str = ''
        msg = (
            'Solver {}{}. '.format(prec_str, kwargs['solver']) +
            'Solve Time {:8.4f} [s]\n'.format(tf - t0)
            )
        print_msg(msg)
    
    u_vec  = merge_vectors(u_intr_vec, bcs_vec, intr_mask)
    u_proj = to_projection(mesh, u_vec)

    return [u_proj, info]

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
