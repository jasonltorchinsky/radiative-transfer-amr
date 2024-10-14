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
import consts
from dg.mesh import Mesh
from dg.projection import Projection
from dg.matrix import get_intr_mask, split_matrix, split_vector, merge_vectors
import utils

# Relative Imports

def solve(self, mesh: Mesh, **kwargs) -> list:
    default_kwargs: dict = {"verbose" : False, # Print info while executing
                            "precondition" : False,
                            "ksp_type" : "gmres", # Which solver to use
                            "pc_type" : "bjacobi", # Which Preconditioner to use
                            "residual_file_path" : None # Plot convergence information to this file path
                            } 
    kwargs: dict = {**default_kwargs, **kwargs}
    
    ## Initialize parallel communicators
    mpi_comm: MPI.Intracomm = MPI.COMM_WORLD
    
    if not PETSc.Sys.isInitialized():
        petsc4py.init(comm = mpi_comm)
    petsc_comm: PETSc.Comm = PETSc.COMM_WORLD
    comm_rank: int = petsc_comm.getRank()
    
    if kwargs["verbose"]:
        if comm_rank == consts.COMM_ROOT:
            ndof: int = mesh.get_ndof()
            ndof: int = mpi_comm.bcast(ndof, root = consts.COMM_ROOT)
        else:
            ndof: int = None
            ndof: int = mpi_comm.bcast(ndof, root = consts.COMM_ROOT)
        msg: str = ( "Initiating solve with {} DoFs...\n".format(ndof) )
        utils.print_msg(msg)
        t0: float = perf_counter()
            
    ## Obtain system matrix
    [ma_sc_mat, _] = self.preconditioner_matrix(mesh, return_preconditioner = False) # Mass - Scattering
    ic_mat: sp._coo.coo_matrix = self.interior_convection_matrix(mesh, **kwargs)
    bc_mat: sp._coo.coo_matrix = self.boundary_convection_matrix(mesh, **kwargs)
    sys_mat: sp._coo.coo_matrix = (bc_mat - ic_mat) + ma_sc_mat
    intr_mask: np.ndarray = get_intr_mask(mesh, **kwargs)
    [sys_mat_intr, sys_mat_bdry] = split_matrix(sys_mat, intr_mask)

    mat_info = sys_mat_intr.getInfo()
        
    ## Obtain forcing vector
    f_vec: np.ndarray = self.forcing_vector(mesh, **kwargs)
    [f_vec_intr, _] = split_vector(f_vec, intr_mask)
    
    ## Obtain boundary conditions vector
    bcs_vec: PETSc.Vec = self.boundary_conditions_vector(mesh, **kwargs)
    [_, bcs_vec_bdry] = split_vector(bcs_vec, intr_mask)

    # Construct (Forcing - BCs)
    rhs_vec: PETSc.Vec = copy.deepcopy(f_vec_intr)
    sys_mat_bdry.mult(bcs_vec_bdry, rhs_vec)
    rhs_vec: PETSc.Vec = f_vec_intr - rhs_vec

    # To get proper split, just copy f_vec_intr
    lhs_vec: np.ndarray = copy.deepcopy(f_vec_intr)
    
    if kwargs["verbose"]:
        t0: float = perf_counter()
        msg: str = ( "Executing solve...\n" )
        utils.print_msg(msg)
        
    # Create the linear system solver
    ksp_type: str = kwargs["ksp_type"]
    ksp: PETSc.KSP = PETSc.KSP()
    ksp.create(comm = petsc_comm)
    ksp.setType(ksp_type)
    ksp.setOperators(sys_mat_intr)
    [rtol, atol, divtol, max_it] = [1.e-9, 1.e-30, 1.e35, 5000]
    GMRESRestart: int = 955
    ksp.setTolerances(rtol   = rtol,   atol   = atol,
                      divtol = divtol, max_it = max_it)
    ksp.setComputeSingularValues(True)
    ksp.setGMRESRestart(GMRESRestart)
    
    pc_type: str = kwargs["pc_type"]
    pc: PETSc.PC = ksp.getPC()
    pc.setType(pc_type)
    
    ksp.setInitialGuessNonzero(True)
    
    ksp.setConvergenceHistory()
    ksp.solve(rhs_vec, lhs_vec)
    PETSc.garbage_cleanup()
    info = ksp.getConvergedReason()
    residuals = ksp.getConvergenceHistory()
    n_iter: int = ksp.getIterationNumber()
    res_f     = ksp.getResidualNorm()
    best_res  = res_f
    best_lhs_vec = copy.deepcopy(lhs_vec)
    
    # If the first solve fails, try try again
    ksp_list = ["qmrcgs", "lgmres", "fbcgsr", "dgmres", "cgs", "pgmres", "gmres",
                "gcr","fgmres"]
    ksp_idx: int = 0
    info = mpi_comm.bcast(info, root = consts.COMM_ROOT)
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
        ksp.create(comm = petsc_comm)
        ksp_type = "none"
        ksp.setType("dgmres")
        ksp.setOperators(sys_mat_intr)
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
        mpi_comm.bcast(info, root = consts.COMM_ROOT)
        residuals = ksp.getConvergenceHistory()
        n_iter    = ksp.getIterationNumber()
        res_f     = ksp.getResidualNorm()
        mpi_comm.barrier()
    elif (info < 0) or (info == 4): # Problem too big to solve directly, just go with the best solution
        lhs_vec = copy.deepcopy(best_lhs_vec)
        
    [emax, emin] = ksp.computeExtremeSingularValues()
    if emin != 0.:
        cond = emax / emin
    else:
        cond = -1.
    ksp.destroy()
    
    if comm_rank == consts.COMM_ROOT:
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
    mpi_comm.barrier()
    
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
        
    uh_vec: np.ndarray = merge_vectors(lhs_vec, bcs_vec_bdry, intr_mask)
    if comm_rank == consts.COMM_ROOT:
        uh_proj: Projection = Projection(mesh)
        uh_proj.from_vector(uh_vec)
    else:
        uh_proj: Projection = None
    mpi_comm.barrier()
        
    return [uh_proj, info, mat_info]
