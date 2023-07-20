import numpy        as np
import petsc4py
import scipy.sparse as sp
from   mpi4py       import MPI
from   petsc4py     import PETSc

def extract_rows_csr(mat, mask):
    """
    Remove the rows denoted by ``indices`` from the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, sp.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    return mat[mask]

def extract_cols_csc(mat, mask):
    """
    Remove the columns denoted by ``indices`` from the CSC sparse matrix ``mat``.
    """
    if not isinstance(mat, sp.csc_matrix):
        raise ValueError("works only for CSC format -- use .tocsc() first")
    return mat[:, mask]

def split_matrix(mesh, mat, intr_mask):
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    comm      = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()

    if comm_rank == 0:
        bdry_mask = np.invert(intr_mask)
        
        mtx = mat.tocsr()
        
        mrows_mtx = extract_rows_csr(mtx, intr_mask)
        mrows_mtx = mrows_mtx.tocsc()
        
        intr_mtx  = extract_cols_csc(mrows_mtx, intr_mask)
        bdry_mtx  = extract_cols_csc(mrows_mtx, bdry_mask)
    else:
        intr_mask = None
        bdry_mtx  = None
    return [intr_mtx, bdry_mtx]
    
def merge_vectors(intr_vec, bdry_vec, intr_mask):
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    comm      = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()

    if comm_rank == 0:
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
    else:
        vec = None
        
    return vec
