import copy
import gc
import inspect
import numpy               as np
import matplotlib.pyplot   as plt
import petsc4py
import psutil
import scipy.sparse        as sp
import scipy.sparse.linalg as spla
import sys
from   mpi4py       import MPI
from   petsc4py     import PETSc
from   inspect      import signature
from   time         import perf_counter

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

    # Set up RNG
    rng = np.random.default_rng()
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    PETSc_comm = PETSc.COMM_WORLD
    comm_rank  = PETSc_comm.getRank()
    comm_size  = PETSc_comm.getSize()
    
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
    #rhs_vec = f_vec_intr.duplicate()
    def zero(x, y, z):
        return 0.
    rhs_vec = calc_forcing_vec(mesh, zero, **kwargs)
    [rhs_vec, _] = mat.split_vector(mesh, rhs_vec, intr_mask)
    M_bdry.mult(bcs_vec_bdry, rhs_vec)
    rhs_vec = f_vec_intr - rhs_vec

    # To get proper split, just copy f_vec_intr
    #lhs_vec = f_vec_intr.duplicate()
    lhs_vec = calc_forcing_vec(mesh, zero, **kwargs)
    [lhs_vec, _] = mat.split_vector(mesh, lhs_vec, intr_mask)
    
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
    [rtol, atol, divtol, max_it] = [1.e-10, 1.e-30, 1.e10, 1000]
    ksp.setTolerances(rtol   = rtol,   atol   = atol,
                      divtol = divtol, max_it = max_it)
    ksp.setComputeSingularValues(True)
    #ksp.setGMRESRestart(3525)
    ksp.setGMRESRestart(155)
    
    pc_type = kwargs['pc_type']
    pc = ksp.getPC()
    pc.setType(pc_type)
    ksp.setInitialGuessNonzero(False)
    
    ksp.setFromOptions()
    
    ksp.setConvergenceHistory()
    ksp.solve(rhs_vec, lhs_vec)
    PETSc.garbage_cleanup()
    MPI_comm.Barrier()
    info = ksp.getConvergedReason()
    residuals = ksp.getConvergenceHistory()
    n_res = np.size(residuals)
    if n_res > 1:
        res_f = residuals[n_res - 1]
    else:
        res_f = -1
    best_res = res_f
    # Have a list of known good solvers/preconditioner combinations in case the one used fails
    ksp_list = ['qmrcgs', 'fbcgsr', 'cgs', 'pgmres', 'dgmres', 'gmres',
                'gcr', 'lgmres', 'fgmres', 'chebyshev']
    #ksp_pc_list = [['qmrcgs', 'kaczmarz'],    ['fbcgsr', 'kaczmarz'], ['cgs', 'kaczmarz'],
    #               ['pgmres', 'kaczmarz'],    ['dgmres', 'kaczmarz'], ['gmres', 'kaczmarz'],
    #               ['gcr', 'kaczmarz'],       ['lgmres', 'kaczmarz'], ['fgmres', 'kaczmarz'],
    #               ['chebyshev', 'kaczmarz'], ['qmrcgs', 'none'],     ['fbcgsr', 'none'],
    #               ['cgs', 'none'],           ['pgmres', 'none'],     ['dgmres', 'none'],
    #               ['gmres', 'none'],         ['gcr', 'none'],        ['lgmres', 'none'],
    #               ['fgmres', 'none'],        ['chebyshev', 'none']]
    #ksp_pc_list = [['richardson', 'jacobi'], ['chebyshev', 'jacobi'], ['gmres', 'jacobi'],
    #               ['fgmres', 'jacobi'], ['lgmres', 'jacobi'], ['dgmres', 'jacobi'],
    #               ['cg', 'jacobi'], ['cgs', 'jacobi'], ['nash', 'jacobi'],
    #               ['qmrcgs', 'jacobi'], ['cr', 'jacobi'], ['gcr', 'jacobi'],
    #               ['pgmres', 'jacobi'], ['richardson', 'pbjacobi'], ['chebyshev', 'pbjacobi'], ['gmres', 'pbjacobi'],
    #               ['fgmres', 'pbjacobi'], ['lgmres', 'pbjacobi'], ['dgmres', 'pbjacobi'],
    #               ['cg', 'pbjacobi'], ['cgs', 'pbjacobi'], ['nash', 'pbjacobi'],
    #               ['qmrcgs', 'pbjacobi'], ['cr', 'pbjacobi'], ['gcr', 'pbjacobi'],
    #               ['pgmres', 'pbjacobi'], ['richardson', 'bjacobi'], ['chebyshev', 'bjacobi'], ['gmres', 'bjacobi'],
    #               ['fgmres', 'bjacobi'], ['lgmres', 'bjacobi'], ['dgmres', 'bjacobi'],
    #               ['cg', 'bjacobi'], ['cgs', 'bjacobi'], ['nash', 'bjacobi'],
    #               ['qmrcgs', 'bjacobi'], ['cr', 'bjacobi'], ['gcr', 'bjacobi'],
    #               ['pgmres', 'bjacobi'], ['richardson', 'kaczmarz'], ['chebyshev', 'kaczmarz'], ['gmres', 'kaczmarz'],
    #               ['fgmres', 'kaczmarz'], ['lgmres', 'kaczmarz'], ['dgmres', 'kaczmarz'],
    #               ['cg', 'kaczmarz'], ['cgs', 'kaczmarz'], ['nash', 'kaczmarz'],
    #               ['qmrcgs', 'kaczmarz'], ['cr', 'kaczmarz'], ['gcr', 'kaczmarz'],
    #               ['pgmres', 'kaczmarz'], ['richardson', 'gamg'], ['chebyshev', 'gamg'], ['gmres', 'gamg'],
    #               ['fgmres', 'gamg'], ['lgmres', 'gamg'], ['dgmres', 'gamg'],
    #               ['cg', 'gamg'], ['cgs', 'gamg'], ['nash', 'gamg'],
    #               ['qmrcgs', 'gamg'], ['cr', 'gamg'], ['gcr', 'gamg'],
    #               ['pgmres', 'gamg']]
    
    ksp_idx = 0
    while (info < 0) or (info == 4): # Solve failed, try something else
        msg = (
            'Iterative solve {} - {} failed.\n'.format(pc_type, ksp_type) +
            12 * ' ' + 'Converged Reason: {}\n'.format(info) +
            12 * ' ' + 'Iteration count:  {}\n'.format(n_res) +
            12 * ' ' + 'Final residual:   {:.4E}\n'.format(res_f) +
            12 * ' ' + 'Best residual:    {:.4E}\n'.format(best_res) +
            12 * ' ' + 'Attempting iterative solve {} - {}\n'.format(pc_type, ksp_list[ksp_idx])
        )
        utils.print_msg(msg)

        # Use the initial guess of the failed solve, because it was ideally at least a little productive
        # Randomly start at zero
        pZero = 0.05
        guessZero = rng.choice((True, False), size = 1, p = (pZero, 1. - pZero))[0]
        if guessZero:
            best_res = 10.**10
            lhs_vec = 0. * lhs_vec
        
        ksp_type = ksp_list[ksp_idx]
        ksp.setType(ksp_type)
        
        PETSc.garbage_cleanup()
        ksp.solve(rhs_vec, lhs_vec)
        PETSc.garbage_cleanup()
        MPI_comm.Barrier()
        info = ksp.getConvergedReason()
        residuals = ksp.getConvergenceHistory()
        n_res = np.size(residuals)
        if n_res > 1:
            res_f = residuals[n_res - 1]
        else:
            res_f = -1
        if res_f > 0.:
            if res_f <= best_res:
                best_res = res_f
        ksp_idx = (ksp_idx+1)%len(ksp_list)
    
    [emax, emin] = ksp.computeExtremeSingularValues()
    if emin != 0.:
        cond = emax / emin
    else:
        cond = -1.
        
    PETSc.garbage_cleanup()
    
    if comm_rank == 0:
        file_path = kwargs['residual_file_path']
        if file_path:
            fig, ax = plt.subplots()
            plt.semilogy(residuals)
            title = '{} - {}, {:.4E}\nConvergence Reason : {}'.format(pc_type, ksp_type, cond, info)
            ax.set_title(title)
            #plt.tight_layout()
            plt.savefig(file_path, dpi = 300)
            plt.close(fig)
    MPI_comm.barrier()
    
    """
    pc_types = ['kaczmarz', 'none']
    ksp_types = ['chebyshev', 'gmres', 'fgmres', 'lgmres', 'dgmres',
                 'cgs', 'qmrcgs',
                 'fbcgsr', 'gcr', 'pgmres']
    for pc_type in pc_types:
        pc.setType(pc_type)
        for ksp_type in ksp_types:
            ksp.setType(ksp_type)
            
            ksp.setFromOptions()
            
            ksp.setConvergenceHistory()
            ksp.solve(rhs_vec, lhs_vec)
            info = ksp.getConvergedReason()
            residuals = ksp.getConvergenceHistory()
            [emax, emin] = ksp.computeExtremeSingularValues()
            if emin != 0.:
                cond = emax / emin
            else:
                cond = -1.
                
            PETSc.garbage_cleanup()

            n_res = np.size(residuals)
            if n_res > 1:
                res_f = residuals[n_res - 1]
            else:
                res_f = -1
            msg = (
                'Solve {} - {} finished.\n'.format(pc_type, ksp_type) +
                12 * ' ' + 'Converged Reason: {}\n'.format(info) +
                12 * ' ' + 'Iteration count:  {}\n'.format(n_res) +
                12 * ' ' + 'Final residual:   {:.4E}\n'.format(res_f) +
                12 * ' ' + 'Condition Number: {:.4E}\n'.format(cond)
            )
            utils.print_msg(msg)
    """
    """
    if info < 0:
        n_res = np.size(residuals)
        if n_res > 1:
            res_f = residuals[n_res - 1]
        else:
            res_f = -1.
        msg = (
            'Iterative solve {} - {} failed.\n'.format(pc_type, ksp_type) +
            12 * ' ' + 'Converged Reason: {}\n'.format(info) +
            12 * ' ' + 'Iteration count:  {}\n'.format(n_res) +
            12 * ' ' + 'Final residual:   {:.4E}\n'.format(res_f) +
            12 * ' ' + 'Condition Number: {:.4E}\n'.format(cond) +
            12 * ' ' + 'Re-attempting iterative solve...\n'
        )
        utils.print_msg(msg)

        if cond <= 5.E2:
            ksp.setGMRESRestart(300)
        else:
            ksp.setGMRESRestart(2750)
        ksp.setFromOptions()
        
        ksp.setConvergenceHistory()
        ksp.solve(rhs_vec, lhs_vec)
        info = ksp.getConvergedReason()
        residuals = ksp.getConvergenceHistory()
        [emax, emin] = ksp.computeExtremeSingularValues()
        cond = emax / emin
        
        PETSc.garbage_cleanup()
        
        if comm_rank == 0:
            file_path = kwargs['residual_file_path']
            if file_path:
                fig, ax = plt.subplots()
                plt.semilogy(residuals)
                title = '{} - {}, {:.4E}'.format(pc_type, ksp_type, cond)
                ax.set_title(title)
                #plt.tight_layout()
                plt.savefig(file_path, dpi = 300)
                plt.close(fig)
        MPI_comm.barrier()
    """
    if kwargs['verbose']:
        tf = perf_counter()
        n_res = np.size(residuals)
        if n_res > 1:
            res_f = residuals[n_res - 1]
        else:
            res_f = -1
        msg = (
            'Solve {} - {} finished.\n'.format(pc_type, ksp_type) +
            12 * ' ' + 'Converged Reason: {}\n'.format(info) +
            12 * ' ' + 'Iteration count:  {}\n'.format(n_res) +
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
        
    return [uh_proj, info]
