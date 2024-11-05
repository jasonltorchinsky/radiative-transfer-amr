# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import petsc4py

from mpi4py   import MPI
from petsc4py import PETSc

# Local Library Imports
import consts

# Relative Imports

def split_matrix(matrix: PETSc.Mat, intr_mask: np.ndarray) -> list:
    ## Initialize parallel communicators
    mpi_comm: MPI.Intracomm = MPI.COMM_WORLD
    
    if not PETSc.Sys.isInitialized():
        petsc4py.init(comm = mpi_comm)
    petsc_comm: PETSc.Comm = PETSc.COMM_WORLD
    
    ## Get global (g_) indexs (idxs) of interior (intr) and boundary (bdry) DoFs
    g_intr_idxs: np.ndarray = np.where(intr_mask)[0].astype(consts.INT)
    g_bdry_idxs: np.ndarray = np.where(np.invert(intr_mask))[0].astype(consts.INT)
    
    ## Broadcast to other ranks
    g_intr_idxs: np.ndarray = mpi_comm.bcast(g_intr_idxs, root = consts.COMM_ROOT)
    g_bdry_idxs: np.ndarray = mpi_comm.bcast(g_bdry_idxs, root = consts.COMM_ROOT)
    
    ## Get local (l_) idxs of intr and bdry DoFs
    ii_0, ii_f = matrix.getOwnershipRange()
    l_intr_mask: np.ndarray = (g_intr_idxs >= ii_0) * (g_intr_idxs < ii_f)
    l_bdry_mask: np.ndarray = (g_bdry_idxs >= ii_0) * (g_bdry_idxs < ii_f)
    
    l_intr_idxs: np.ndarray = g_intr_idxs[l_intr_mask]
    l_bdry_idxs: np.ndarray = g_bdry_idxs[l_bdry_mask]

    intr_IS: PETSc.IS = PETSc.IS()
    intr_IS.createGeneral(list(l_intr_idxs), comm = petsc_comm)
    
    bdry_IS: PETSc.IS = PETSc.IS()
    bdry_IS.createGeneral(list(l_bdry_idxs), comm = petsc_comm)
    
    ## Get intr and bdry parts of matrix
    matrix_intr: PETSc.Mat = matrix.createSubMatrix(isrow = intr_IS, iscol = intr_IS)
    matrix_bdry: PETSc.Mat = matrix.createSubMatrix(isrow = intr_IS, iscol = bdry_IS)
    
    return [matrix_intr, matrix_bdry]