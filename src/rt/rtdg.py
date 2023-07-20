import numpy        as     np
import petsc4py
import scipy.sparse as     sp
import sys
from   mpi4py       import MPI
from   petsc4py     import PETSc
from   inspect      import signature
from   time         import perf_counter

import dg.matrix     as mat
import dg.projection as proj
import utils

from .calc_precond_matrix   import calc_precond_matrix
from .calc_intr_conv_matrix import calc_intr_conv_matrix
from .calc_bdry_conv_matrix import calc_bdry_conv_matrix
from .calc_forcing_vec      import calc_forcing_vec
from .calc_bcs_vec          import calc_bcs_vec

def rtdg(mesh, kappa, sigma, Phi, bcs_dirac, f = None, **kwargs):
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
        t0  = perf_counter()
        msg = (
            'Initiating solve with {} DoFs...\n'.format(mesh.get_ndof())
            )
        utils.print_msg(msg)
            
    # Calculate A, b of Ax=b in serial
    [M_mass_scat, M_pc] = calc_precond_matrix(mesh, kappa, sigma, Phi, **kwargs)
    M_intr_conv         = calc_intr_conv_matrix(mesh, **kwargs)
    M_bdry_conv         = calc_bdry_conv_matrix(mesh, **kwargs)
    M_conv = M_bdry_conv - M_intr_conv
    M = M_conv + M_mass_scat
    intr_mask = mat.get_intr_mask(mesh)
    [M_intr, M_bdry] = mat.split_matrix(mesh, M, intr_mask)
    
    if kwargs['precondition']:
        [M_pc, _] = mat.split_matrix(mesh, M_pc, intr_mask)
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
    
    if comm_rank == 0:
        f_vec = calc_forcing_vec(mesh, forcing)
        f_intr_vec = f_vec[intr_mask]

        M_global = M_intr
        f_global = f_intr_vec - M_bdry @ bcs_vec

        n_global = np.size(f_global)
    else:
        n_global = None

    MPI_comm.Barrier()
    n_global = MPI_comm.bcast(n_global, root = 0)
    M_local = None
    f_local = None
    
    ## Convert the system to parallel
    if kwargs['verbose']:
        t0 = perf_counter()
        msg = (
            'Setting up and performing solve...\n'
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
    (II, _, VV) = sp.find(f_local)
    nnz_local    = np.size(II)
    for idx in range(0, nnz_local):
        ii = II[idx]
        vv = VV[idx]
        
        f_MPI[ii + ii_0] = vv
        
    f_MPI.assemblyBegin()
    f_MPI.assemblyEnd()
    
    # Create the linear system solver
    ksp = PETSc.KSP()
    ksp.create(comm = comm)
    ksp.setType('lgmres')
    ksp.setOperators(M_MPI)
    ksp.setFromOptions()
    
    ksp.solve(f_MPI, u_MPI)
    
    u_local = u_MPI[ii_0:ii_f]
    u_global = MPI_comm.gather(u_local, root = 0)
    if comm_rank == 0:
        u_intr_vec = np.concatenate(u_global)
    
    if kwargs['verbose']:
        tf = perf_counter()
        msg = (
            'Solve Completed.\n' +
            12 * ' ' + 'Time Elapsed: {:8.4f} [s]\n'.format(tf - t0)
            )
        utils.print_msg(msg)
            
    if comm_rank == 0:
        u_vec  = mat.merge_vectors(u_intr_vec, bcs_vec, intr_mask)
        u_proj = proj.to_projection(mesh, u_vec)
    else:
        u_proj = None
        
    u_proj = MPI_comm.bcast(u_proj, root = 0)
    info = 0
    
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
