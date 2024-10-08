# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import petsc4py

from mpi4py   import MPI
from petsc4py import PETSc

# Local Library Imports

# Relative Imports

def merge_vectors(vector_intr: PETSc.Vec, vector_bdry: PETSc.Vec,
                  intr_mask: np.ndarray) -> np.ndarray:
    ## Initialize parallel communicators
    mpi_comm: MPI.Intracomm = MPI.COMM_WORLD
    
    if not PETSc.Sys.isInitialized():
        petsc4py.init(comm = mpi_comm)
    petsc_comm: PETSc.Comm = PETSc.COMM_WORLD
    comm_rank: int = petsc_comm.getRank()

    root_rank: int = 0
    
    intr_ii_0, intr_ii_f = vector_intr.getOwnershipRange()
    g_intr_vec: np.ndarray = mpi_comm.gather(vector_intr[intr_ii_0:intr_ii_f],
                                             root = root_rank)
    
    bdry_ii_0, bdry_ii_f = vector_bdry.getOwnershipRange()
    g_bdry_vec: np.ndarray = mpi_comm.gather(vector_bdry[bdry_ii_0:bdry_ii_f],
                                             root = root_rank)
    
    if comm_rank == root_rank:
        g_intr_vec: np.ndarray = np.concatenate(g_intr_vec)
        g_bdry_vec: np.ndarray = np.concatenate(g_bdry_vec)
        
        ndof: int = np.size(intr_mask)
        
        vector: np.ndarray = np.zeros(ndof)
        intr_idx: int = 0
        bdry_idx: int = 0
        
        for ii in range(0, ndof):
            if intr_mask[ii]:
                vector[ii] = g_intr_vec[intr_idx]
                
                intr_idx += 1
            else:
                vector[ii] = g_bdry_vec[bdry_idx]
                
                bdry_idx += 1
    else:
        vector: np.ndarray = np.empty(0)
        
    return vector