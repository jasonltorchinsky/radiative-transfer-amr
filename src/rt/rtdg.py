"""
Returns the solution.
"""

# Standard Library Imports
import copy
import gc
import inspect
import sys
from   inspect      import signature
from   time         import perf_counter

# Third-Party Library Imports
import numpy               as np
import matplotlib.pyplot   as plt
import petsc4py
import psutil
import scipy.sparse        as sp
import scipy.sparse.linalg as spla
from   mpi4py       import MPI
from   petsc4py     import PETSc

# Local Library Imports
import dg.matrix     as mat
import dg.projection as proj
import utils

from .calc_bcs_vec          import calc_bcs_vec
from .calc_bdry_conv_matrix import calc_bdry_conv_matrix
from .calc_forcing_vec      import calc_forcing_vec
from .calc_intr_conv_matrix import calc_intr_conv_matrix
from .calc_mass_matrix      import calc_mass_matrix
from .calc_precond_matrix   import calc_precond_matrix
from .calc_scat_matrix      import calc_scat_matrix

def rtdg(mesh, kappa, sigma, Phi, bcs_dirac, f = None, **kwargs):
    return rtdg_mpi(mesh, kappa, sigma, Phi, bcs_dirac, f, **kwargs)
def rtdg_mpi(mesh, kappa, sigma, Phi, bcs_dirac, f = None, **kwargs):
    """
    Solve the two-dimensional radiative transfer model.
    """
    
    default_kwargs = {'verbose'      : False, # Print info while executing
                      'precondition' : False,
                      'ksp_type' : 'gmres', # Which solver to use
                      'pc_type' : 'bjacobi',  # Which Preconditioner to use
                      'residual_file_path' : None # Plot convergence information to this file path
                      } 
    kwargs = {**default_kwargs, **kwargs}
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    PETSc_comm = PETSc.COMM_WORLD
    comm_rank  = PETSc_comm.getRank()
    comm_size  = PETSc_comm.getSize()
    
    # Set up RNG
    rng = np.random.default_rng()
    
    if kwargs['verbose']:
        if comm_rank == 0:
            ndof = mesh.get_ndof()
            ndof = MPI_comm.bcast(ndof, root = 0)
        else:
            ndof = None
            ndof = MPI_comm.bcast(ndof, root = 0)
        msg = (
            'Initiating solve with {} DoFs...\n'.format(ndof)
            )
        utils.print_msg(msg)
        t0  = perf_counter()
            
    # Calculate
    [M_mass_scat, _] = calc_precond_matrix(mesh, kappa, sigma, Phi, **kwargs)
    M_intr_conv = calc_intr_conv_matrix(mesh, **kwargs)
    M_bdry_conv = calc_bdry_conv_matrix(mesh, **kwargs)
    M_conv = M_bdry_conv - M_intr_conv
    M      = M_conv + M_mass_scat
    intr_mask = mat.get_intr_mask(mesh, **kwargs)
    [M_intr, M_bdry] = mat.split_matrix(mesh, M, intr_mask)

    mat_info = M_intr.getInfo()
    
    if kwargs['precondition']:
        [M_pc, _] = mat.split_matrix(mesh, M_pc, intr_mask)
    else:
        M_pc = None
        
    # Make sure forcing function takes three arguments
    if f is None:
        def forcing(x, y, th):
            return 0
    if len(inspect.signature(f).parameters) == 1:
        def forcing(x, y, th):
            return f(x)
    elif len(inspect.signature(f).parameters) == 2:
        def forcing(x, y, th):
            return f(x, y)
    elif len(inspect.signature(f).parameters) == 3:
        def forcing(x, y, th):
            return f(x, y, th)
    f_vec           = calc_forcing_vec(mesh, forcing, **kwargs)
    [f_vec_intr, _] = mat.split_vector(mesh, f_vec, intr_mask)
    
    bcs_vec           = calc_bcs_vec(mesh, bcs_dirac, **kwargs)
    [_, bcs_vec_bdry] = mat.split_vector(mesh, bcs_vec, intr_mask)

    # Construct (Forcing - BCs)
    rhs_vec = copy.deepcopy(f_vec_intr)
    M_bdry.mult(bcs_vec_bdry, rhs_vec)
    rhs_vec = f_vec_intr - rhs_vec

    # To get proper split, just copy f_vec_intr
    lhs_vec = copy.deepcopy(f_vec_intr)
    
    
    if kwargs['verbose']:
        t0 = perf_counter()
        msg = (
            'Executing solve...\n'
        )
        utils.print_msg(msg)
        
    # Create the linear system solver
    ksp_type = kwargs['ksp_type']
    ksp = PETSc.KSP()
    ksp.create(comm = PETSc_comm)
    ksp.setType(ksp_type)
    ksp.setOperators(M_intr)
    [rtol, atol, divtol, max_it] = [1.e-9, 1.e-30, 1.e35, 5000]
    GMRESRestart = 955
    ksp.setTolerances(rtol   = rtol,   atol   = atol,
                      divtol = divtol, max_it = max_it)
    ksp.setComputeSingularValues(True)
    ksp.setGMRESRestart(GMRESRestart)
    
    pc_type = kwargs['pc_type']
    pc = ksp.getPC()
    pc.setType(pc_type)
    
    ksp.setInitialGuessNonzero(True)
    
    ksp.setConvergenceHistory()
    ksp.solve(rhs_vec, lhs_vec)
    PETSc.garbage_cleanup()
    info = ksp.getConvergedReason()
    residuals = ksp.getConvergenceHistory()
    n_iter    = ksp.getIterationNumber()
    res_f     = ksp.getResidualNorm()
    best_res  = res_f
    best_lhs_vec = copy.deepcopy(lhs_vec)
    
    # If the first solve fails, try try again
    ksp_list = ['qmrcgs', 'lgmres', 'fbcgsr', 'dgmres', 'cgs', 'pgmres', 'gmres',
                'gcr','fgmres']
    ksp_idx = 0
    info = MPI_comm.bcast(info, root = 0)
    nsolve = 0
    while False and ((info < 0) or (info == 4) and nsolve < 25): # Solve failed, try something else
        ksp.destroy()
        MPI_comm.barrier()
        msg = (
            'Iterative solve {} - {} failed.\n'.format(pc_type, ksp_type) +
            12 * ' ' + 'Converged Reason: {}\n'.format(info) +
            12 * ' ' + 'Iteration count:  {}\n'.format(n_iter) +
            12 * ' ' + 'Final residual:   {:.4E}\n'.format(res_f) +
            12 * ' ' + 'Best residual:    {:.4E}\n'.format(best_res) +
            12 * ' ' + 'Attempting iterative solve {} - {}\n'.format(pc_type, ksp_list[ksp_idx])
        )
        utils.print_msg(msg)

        # Use the initial guess of the failed solve, because it was ideally at least a little productive
        # Randomly start at zero
        pZero = 0.05
        guessZero = rng.choice((True, False), size = 1, p = (pZero, 1. - pZero))[0]
        MPI_comm.bcast(guessZero, root = 0)
        if guessZero or best_res > 10.**2:
            best_res = 10.**10
            lhs_vec = 0. * lhs_vec
        else:
            lhs_vec = copy.deepcopy(best_lhs_vec)
        MPI_comm.barrier()
        ksp_type = ksp_list[ksp_idx]
        ksp = PETSc.KSP()
        ksp.create(comm = PETSc_comm)
        ksp.setType(ksp_type)
        ksp.setOperators(M_intr)
        ksp.setTolerances(rtol   = rtol,   atol   = atol,
                          divtol = divtol, max_it = max_it)
        ksp.setComputeSingularValues(True)
        ksp.setGMRESRestart(GMRESRestart)
        
        pc_type = kwargs['pc_type']
        pc = ksp.getPC()
        pc.setType(pc_type)
        
        ksp.setInitialGuessNonzero(True) # Might need to be true if crashes after morning of Aug. 10
        
        ksp.setConvergenceHistory()
        ksp.solve(rhs_vec, lhs_vec)
        PETSc.garbage_cleanup()
        info = ksp.getConvergedReason()
        MPI_comm.bcast(info, root = 0)
        residuals = ksp.getConvergenceHistory()
        n_iter    = ksp.getIterationNumber()
        res_f     = ksp.getResidualNorm()
        MPI_comm.bcast(res_f, root = 0)
        if res_f > 0.:
            if res_f <= best_res:
                best_res = res_f
                best_lhs_vec = copy.deepcopy(lhs_vec)
        ksp_idx = (ksp_idx+1)%len(ksp_list)
        nsolve += 1
        MPI_comm.barrier()
        
    # If the system is fairly small and the interative solves failed, try a direct solve
    if ((info < 0) or (info == 4)) and (ndof < 1.2e5):
        ksp.destroy()
        msg = (
            'Iterative solve {} - {} failed.\n'.format(pc_type, ksp_type) +
            12 * ' ' + 'Converged Reason: {}\n'.format(info) +
            12 * ' ' + 'Iteration count:  {}\n'.format(n_iter) +
            12 * ' ' + 'Final residual:   {:.4E}\n'.format(res_f) +
            12 * ' ' + 'Best residual:    {:.4E}\n'.format(best_res) +
            12 * ' ' + 'Attempting direct LU solve\n'.format(pc_type, ksp_list[ksp_idx])
        )
        utils.print_msg(msg)
        
        ksp = PETSc.KSP()
        ksp.create(comm = PETSc_comm)
        ksp_type = 'none'
        ksp.setType('dgmres')
        ksp.setOperators(M_intr)
        ksp.setTolerances(rtol   = rtol,   atol   = atol,
                          divtol = divtol, max_it = max_it)
        ksp.setComputeSingularValues(True)
        ksp.setGMRESRestart(GMRESRestart)
        
        pc = ksp.getPC()
        pc_type = 'lu'
        pc.setType(pc_type)
        
        ksp.setInitialGuessNonzero(False)
        
        ksp.setConvergenceHistory()
        ksp.solve(rhs_vec, lhs_vec)
        PETSc.garbage_cleanup()
        info = ksp.getConvergedReason()
        MPI_comm.bcast(info, root = 0)
        residuals = ksp.getConvergenceHistory()
        n_iter    = ksp.getIterationNumber()
        res_f     = ksp.getResidualNorm()
        MPI_comm.barrier()
    elif (info < 0) or (info == 4): # Problem too big to solve directly, just go with the best solution
        lhs_vec = copy.deepcopy(best_lhs_vec)
        
    [emax, emin] = ksp.computeExtremeSingularValues()
    if emin != 0.:
        cond = emax / emin
    else:
        cond = -1.
    ksp.destroy()
    
    if comm_rank == 0:
        file_path = kwargs['residual_file_path']
        if file_path:
            fig, ax = plt.subplots()
            plt.semilogy(residuals)
            plt.axhline(rtol)
            title = '{} - {}, {:.4E}\nConvergence Reason : {}'.format(pc_type, ksp_type, cond, info)
            ax.set_title(title)
            #plt.tight_layout()
            plt.savefig(file_path, dpi = 300)
            plt.close(fig)
    MPI_comm.barrier()
    
    if kwargs['verbose']:
        tf = perf_counter()
        msg = (
            'Solve {} - {} finished.\n'.format(pc_type, ksp_type) +
            12 * ' ' + 'Converged Reason: {}\n'.format(info) +
            12 * ' ' + 'Iteration count:  {}\n'.format(n_iter) +
            12 * ' ' + 'Final residual:   {:.4E}\n'.format(res_f) +
            12 * ' ' + 'Condition Number: {:.4E}\n'.format(cond) +
            12 * ' ' + 'Time Elapsed:   {:8.4f} [s]\n'.format(tf - t0)
        )
        utils.print_msg(msg)
        
    uh_vec  = mat.merge_vectors(lhs_vec, bcs_vec_bdry, intr_mask)
    if comm_rank == 0:
        uh_proj = proj.to_projection(mesh, uh_vec)
    else:
        uh_proj = None
    MPI_comm.barrier()
        
    return [uh_proj, info, mat_info]
