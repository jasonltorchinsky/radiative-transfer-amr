# Standard Library Imports
import copy
from time import perf_counter

# Third-Party Library Imports
import numpy as np
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
    default_kwargs: dict = {"verbose"  : False, # Print info while executing
                            "ksp_type" : "gmres", # Which solver to use
                            "pc_type"  : "bjacobi" # Which Preconditioner to use
                            } 
    kwargs: dict = {**default_kwargs, **kwargs}
    
    ## Initialize parallel communicators
    mpi_comm: MPI.Intracomm = MPI.COMM_WORLD
    
    if not PETSc.Sys.isInitialized():
        petsc4py.init(comm = mpi_comm)
    petsc_comm: PETSc.Comm = PETSc.COMM_WORLD
    comm_rank: int = petsc_comm.getRank()
    
    if comm_rank == consts.COMM_ROOT:
        ndof: int = mesh.get_ndof()
        ndof: int = mpi_comm.bcast(ndof, root = consts.COMM_ROOT)
    else:
        ndof: int = None
        ndof: int = mpi_comm.bcast(ndof, root = consts.COMM_ROOT)

    if kwargs["verbose"]:
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

    matrix_info: dict = sys_mat_intr.getInfo()
        
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
    [rtol, atol, divtol, max_it] = [consts.EPS, consts.EPS,
                                    np.sqrt(consts.INF), 5000]
    GMRESRestart: int = 50
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

    ## Get the convergence information
    converged_reason: int = ksp.getConvergedReason()
    convergence_history: np.ndarray = ksp.getConvergenceHistory()
    iteration_number: int = ksp.getIterationNumber()
    res_f: float = ksp.getResidualNorm()
    res_best: float = np.min(convergence_history)
        
    [emax, emin] = ksp.computeExtremeSingularValues()
    matrix_info["extreme_singular_values"] = [emax, emin]
    if emin != 0.:
        cond: float = emax / emin
    else:
        cond: float = consts.INF

    ksp.destroy()
    
    if kwargs["verbose"]:
        tf: float = perf_counter()
        msg: str = (
            "Solve {} - {} finished.\n".format(pc_type, ksp_type) +
            12 * " " + "Converged Reason: {}\n".format(converged_reason) +
            12 * " " + "Iteration count:  {}\n".format(iteration_number) +
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

    convergence_info: dict = {"converged_reason" : converged_reason,
                              "convergence_history" : convergence_history.tolist(),
                              "iteration_number" : iteration_number,
                              "res_f" : res_f,
                              "res_best" : res_best,
                              "rtol" : rtol,
                              "atol" : atol,
                              "divtol" : divtol,
                              "max_it" : max_it}
        
    return [uh_proj, convergence_info, matrix_info]
