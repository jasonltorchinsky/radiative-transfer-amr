# Standard Library Imports
import copy
import inspect
from inspect import signature
from time import perf_counter
from typing import Callable

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt
import petsc4py
import scipy.sparse as sp
from mpi4py import MPI
from petsc4py import PETSc

# Local Library Imports
from dg.mesh import Mesh
from dg.projection import Projection
from dg.matrix import get_intr_mask, split_matrix, split_vector, merge_vectors
import utils

from .calc_bcs_vec          import calc_bcs_vec
from .calc_bdry_conv_matrix import calc_bdry_conv_matrix
from .calc_forcing_vec      import calc_forcing_vec
from .calc_intr_conv_matrix import calc_intr_conv_matrix
from .calc_precond_matrix   import calc_precond_matrix

def rtdg(mesh: Mesh,
         kappa: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], 
         sigma: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
         Phi: Callable[[np.ndarray, np.ndarray], np.ndarray],
         bcs_dirac: list, f = None, **kwargs) -> list:
    """
    Solve the two-dimensional radiative transfer model.
    """

    default_kwargs: dict = {"verbose" : False, # Print info while executing
                            "precondition" : False,
                            "ksp_type" : "gmres", # Which solver to use
                            "pc_type" : "bjacobi",  # Which Preconditioner to use
                            "residual_file_path" : None # Plot convergence information to this file path
                            } 
    kwargs: dict = {**default_kwargs, **kwargs}
    
    # Initialize parallel communicators
    MPI_comm: MPI.Intracomm = MPI.COMM_WORLD
    
    if not PETSc.Sys.isInitialized():
        petsc4py.init(comm = MPI_comm)
    PETSc_comm: PETSc.Comm = PETSc.COMM_WORLD
    comm_rank: int = PETSc_comm.getRank()
    
    if kwargs["verbose"]:
        if comm_rank == 0:
            ndof: int = mesh.get_ndof()
            ndof: int = MPI_comm.bcast(ndof, root = 0)
        else:
            ndof: int = None
            ndof: int = MPI_comm.bcast(ndof, root = 0)
        msg: str = ( "Initiating solve with {} DoFs...\n".format(ndof) )
        utils.print_msg(msg)
        t0: float = perf_counter()
            
    # Calculate
    [M_mass_scat, _] = calc_precond_matrix(mesh, kappa, sigma, Phi, **kwargs)
    M_intr_conv: sp._coo.coo_matrix = calc_intr_conv_matrix(mesh, **kwargs)
    M_bdry_conv: sp._coo.coo_matrix = calc_bdry_conv_matrix(mesh, **kwargs)
    M_conv: sp._coo.coo_matrix = M_bdry_conv - M_intr_conv
    M: sp._coo.coo_matrix = M_conv + M_mass_scat
    intr_mask: np.ndarray = get_intr_mask(mesh, **kwargs)
    [M_intr, M_bdry] = split_matrix(mesh, M, intr_mask)

    mat_info = M_intr.getInfo()
    
    if kwargs["precondition"]:
        [M_pc, _] = split_matrix(mesh, M_pc, intr_mask)
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
    f_vec: np.ndarray = calc_forcing_vec(mesh, forcing, **kwargs)
    [f_vec_intr, _] = split_vector(mesh, f_vec, intr_mask)
    
    bcs_vec: PETSc.Vec = calc_bcs_vec(mesh, bcs_dirac, **kwargs)
    [_, bcs_vec_bdry] = split_vector(mesh, bcs_vec, intr_mask)

    # Construct (Forcing - BCs)
    rhs_vec = copy.deepcopy(f_vec_intr)
    M_bdry.mult(bcs_vec_bdry, rhs_vec)
    rhs_vec = f_vec_intr - rhs_vec

    # To get proper split, just copy f_vec_intr
    lhs_vec = copy.deepcopy(f_vec_intr)
    
    
    if kwargs["verbose"]:
        t0: float = perf_counter()
        msg: str = ( "Executing solve...\n" )
        utils.print_msg(msg)
        
    # Create the linear system solver
    ksp_type = kwargs["ksp_type"]
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
    
    pc_type = kwargs["pc_type"]
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
    ksp_list = ["qmrcgs", "lgmres", "fbcgsr", "dgmres", "cgs", "pgmres", "gmres",
                "gcr","fgmres"]
    ksp_idx = 0
    info = MPI_comm.bcast(info, root = 0)
    nsolve = 0
    # If the system is fairly small and the interative solves failed, try a direct solve
    if ((info < 0) or (info == 4)) and (ndof < 1.2e5):
        ksp.destroy()
        msg = (
            "Iterative solve {} - {} failed.\n".format(pc_type, ksp_type) +
            12 * " " + "Converged Reason: {}\n".format(info) +
            12 * " " + "Iteration count:  {}\n".format(n_iter) +
            12 * " " + "Final residual:   {:.4E}\n".format(res_f) +
            12 * " " + "Best residual:    {:.4E}\n".format(best_res) +
            12 * " " + "Attempting direct LU solve\n".format(pc_type, ksp_list[ksp_idx])
        )
        utils.print_msg(msg)
        
        ksp = PETSc.KSP()
        ksp.create(comm = PETSc_comm)
        ksp_type = "none"
        ksp.setType("dgmres")
        ksp.setOperators(M_intr)
        ksp.setTolerances(rtol   = rtol,   atol   = atol,
                          divtol = divtol, max_it = max_it)
        ksp.setComputeSingularValues(True)
        ksp.setGMRESRestart(GMRESRestart)
        
        pc = ksp.getPC()
        pc_type = "lu"
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
        file_path = kwargs["residual_file_path"]
        if file_path:
            fig, ax = plt.subplots()
            plt.semilogy(residuals)
            plt.axhline(rtol)
            title = "{} - {}, {:.4E}\nConvergence Reason : {}".format(pc_type, ksp_type, cond, info)
            ax.set_title(title)
            #plt.tight_layout()
            plt.savefig(file_path, dpi = 300)
            plt.close(fig)
    MPI_comm.barrier()
    
    if kwargs["verbose"]:
        tf = perf_counter()
        msg = (
            "Solve {} - {} finished.\n".format(pc_type, ksp_type) +
            12 * " " + "Converged Reason: {}\n".format(info) +
            12 * " " + "Iteration count:  {}\n".format(n_iter) +
            12 * " " + "Final residual:   {:.4E}\n".format(res_f) +
            12 * " " + "Condition Number: {:.4E}\n".format(cond) +
            12 * " " + "Time Elapsed:   {:8.4f} [s]\n".format(tf - t0)
        )
        utils.print_msg(msg)
        
    uh_vec = merge_vectors(lhs_vec, bcs_vec_bdry, intr_mask)
    if comm_rank == 0:
        uh_proj: Projection = Projection(mesh)
        uh_proj.from_vector(uh_vec)
    else:
        uh_proj: Projection = None
    MPI_comm.barrier()
        
    return [uh_proj, info, mat_info]
