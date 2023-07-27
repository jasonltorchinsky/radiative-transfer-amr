import copy
import gc
import inspect
import numpy               as np
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
                      'precondition' : False  # Calculate PC matrix
                      } 
    kwargs = {**default_kwargs, **kwargs}
    
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
    M_intr_conv      = calc_intr_conv_matrix(mesh, **kwargs)
    M_bdry_conv      = calc_bdry_conv_matrix(mesh, **kwargs)
    M_conv    = M_bdry_conv - M_intr_conv
    M         = M_conv + M_mass_scat
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
    f_vec      = calc_forcing_vec(mesh, forcing, **kwargs)
    [f_vec_intr, _] = mat.split_vector(mesh, f_vec, intr_mask)
    
    bcs_vec    = calc_bcs_vec(mesh, bcs_dirac, **kwargs)
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
    ksp = PETSc.KSP()
    ksp.create(comm = PETSc_comm)
    ksp.setType('lgmres')
    ksp.setOperators(M_intr)
    ksp.setTolerances(rtol = 1.e-10, atol = 1.e-50, divtol = 1.e7)
    
    pc = ksp.getPC()
    pc.setType('lu')
    
    ksp.setFromOptions()
    
    ksp.solve(rhs_vec, lhs_vec)
    info = ksp.getConvergedReason()
    
    if kwargs['verbose']:
        tf = perf_counter()
        msg = (
            'Solve Completed. Converged Reason: {}\n'.format(info) +
            12 * ' ' + 'Time Elapsed: {:8.4f} [s]\n'.format(tf - t0)
        )
        utils.print_msg(msg)
        
    uh_vec  = mat.merge_vectors(lhs_vec, bcs_vec_bdry, intr_mask)
    if comm_rank == 0:
        uh_proj = proj.to_projection(mesh, uh_vec)
    else:
        uh_proj = None
        
    return [uh_proj, info]

def rtdg_seq(mesh, kappa, sigma, Phi, bcs_dirac, f = None, **kwargs):
    """
    Solve the two-dimensional radiative transfer model.
    """
    
    default_kwargs = {'verbose'      : False, # Print info while executing
                      'precondition' : False  # Calculate PC matrix
                      } 
    kwargs = {**default_kwargs, **kwargs}

    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    comm      = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()
    
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
            
    # Calculate A, b of Ax=b in serial
    [M_mass_scat, M_pc] = calc_precond_matrix(mesh, kappa, sigma, Phi, **kwargs)
    M_intr_conv         = calc_intr_conv_matrix(mesh, **kwargs)
    M_bdry_conv         = calc_bdry_conv_matrix(mesh, **kwargs)
    M_conv    = M_bdry_conv - M_intr_conv
    M         = M_conv + M_mass_scat
    intr_mask = mat.get_intr_mask(mesh, **kwargs)
    [M_intr, M_bdry] = mat.split_matrix(mesh, M, intr_mask)
    
    if kwargs['precondition']:
        [M_pc, _] = mat.split_matrix(mesh, M_pc, intr_mask)
    else:
        M_pc = None
        
    # If using too much memory, delete used matrices
    used_mem = psutil.virtual_memory()[2]
    if used_mem >= 80:
        del M_mass_scat, M_intr_conv, M_bdry_conv, M
        gc.collect()
        
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
    bcs_vec    = calc_bcs_vec(mesh, bcs_dirac, **kwargs)
    bdry_mask  = np.invert(intr_mask)
    f_vec      = calc_forcing_vec(mesh, forcing, **kwargs)
    f_intr_vec = f_vec[intr_mask]
    
    if comm_rank == 0:
        M_global = M_intr
        f_global = f_intr_vec - M_bdry @ bcs_vec
        
        n_global = np.size(f_global)
    else:
        n_global = None
        
    # If using too much memory, delete used variables
    used_mem = psutil.virtual_memory()[2]
    if used_mem >= 80:
        del f_vec, f_intr_vec
        gc.collect()
        
    MPI_comm.Barrier()

    if comm_size > 0:
        n_global = MPI_comm.bcast(n_global, root = 0)
        M_local = None
        f_local = None
        
        ## Convert the system to parallel
        if kwargs['verbose']:
            t0 = perf_counter()
            msg = (
                'Setting up solve...\n'
            )
            utils.print_msg(msg)
            
        # Create PETSc sparse matrix
        M_MPI = PETSc.Mat()
        M_MPI.createAIJ(size = [n_global, n_global], comm = comm)
        
        o_rngs = M_MPI.getOwnershipRanges()
        ii_0 = o_rngs[comm_rank]
        ii_f = o_rngs[comm_rank+1]
        if comm_rank == 0:
            # Communicate global information
            for rank in range(1, comm_size):
                ii_0_else = o_rngs[rank]
                ii_f_else = o_rngs[rank+1]
                MPI_comm.send(M_global[ii_0_else:ii_f_else, :],
                              dest = rank)
            M_local = M_global[ii_0:ii_f, :]
            
            MPI_comm.Barrier()
            
            for rank in range(1, comm_size):
                ii_0_else = o_rngs[rank]
                ii_f_else = o_rngs[rank+1]
                MPI_comm.send(f_global[ii_0_else:ii_f_else],
                              dest = rank)
                f_local = f_global[ii_0:ii_f]
        else:
            M_local = MPI_comm.recv(source = 0)
            
            MPI_comm.Barrier()
            
            f_local = MPI_comm.recv(source = 0)
            
        # Put A_local into the shared matrix
        (II, JJ, VV) = sp.find(M_local)
        nnz_local    = np.size(II)
        for idx in range(0, nnz_local):
            ii = II[idx]
            jj = JJ[idx]
            vv = VV[idx]
            
            M_MPI[ii + ii_0, jj] = vv
            
        # Communicate off-rank values and setup internal data structures for
        # performing parallel operations
        M_MPI.assemblyBegin()
        M_MPI.assemblyEnd()
        
        # Create LHS, RHS vectors and fill them in
        u_MPI, f_MPI = M_MPI.createVecs()
        u_MPI.set(0)
        (_, II, VV) = sp.find(f_local)
        nnz_local   = np.size(II)
        for idx in range(0, nnz_local):
            ii = II[idx]
            vv = VV[idx]
            
            f_MPI[ii + ii_0] = vv
            
        f_MPI.assemblyBegin()
        f_MPI.assemblyEnd()
        
        if kwargs['verbose']:
            tf = perf_counter()
            msg = (
                'Solve set up.\n' +
                12 * ' ' + 'Time Elapsed: {:8.4f} [s]\n'.format(tf - t0)
            )
            utils.print_msg(msg)
            
        if kwargs['verbose']:
            t0 = perf_counter()
            msg = (
                'Executing solve...\n'
            )
            utils.print_msg(msg)
            
        # Create the linear system solver
        ksp = PETSc.KSP()
        ksp.create(comm = comm)
        ksp.setType('gmres')
        ksp.setOperators(M_MPI)
        ksp.setTolerances(rtol = 1.e-10, atol   = 1.e-50,
                          divtol = 1.e7)
        
        pc = ksp.getPC()
        pc.setType('lu')
        
        ksp.setFromOptions()
        
        ksp.solve(f_MPI, u_MPI)
        info = ksp.getConvergedReason()
        
        # If using too much memory, delete used variables
        used_mem = psutil.virtual_memory()[2]
        if used_mem >= 80:
            del ksp, pc, f_MPI, M_MPI
            gc.collect()
            
        if info > 0:
            u_local = u_MPI[ii_0:ii_f]
            u_global = MPI_comm.gather(u_local, root = 0)
            
            if comm_rank == 0:
                u_intr_vec = np.concatenate(u_global)
        else: # Iterative solver failed, go to direct solve
            msg = (
                'Parallel iterative solve failed: {}. Attempting direct solve...\n'.format(info)
            )
            utils.print_msg(msg)
            if comm_rank == 0:
                u_intr_vec = spla.spsolve(M_global, f_global)
                
        if kwargs['verbose']:
            tf = perf_counter()
            msg = (
                'Solve Completed. Exit Code: {}\n'.format(info) +
                12 * ' ' + 'Time Elapsed: {:8.4f} [s]\n'.format(tf - t0)
            )
            utils.print_msg(msg)
            
        if comm_rank == 0:
            u_vec  = mat.merge_vectors(u_intr_vec, bcs_vec, intr_mask)
            u_proj = proj.to_projection(mesh, u_vec)
        else:
            u_proj = None
    else:
        if comm_rank == 0:
            u_intr_vec = spla.spsolve(M_global, f_global)
            u_vec  = mat.merge_vectors(u_intr_vec, bcs_vec, intr_mask)
            u_proj = proj.to_projection(mesh, u_vec)
            info   = 0
        else:
            u_proj = None
            info   = 0
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
