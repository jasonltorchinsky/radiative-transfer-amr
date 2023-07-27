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
    return split_matrix_mpi(mesh, mat, intr_mask)

def split_matrix_seq(mesh, mat, intr_mask):
    
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
        
        intr_mtx  = extract_cols_csc(mrows_mtx, intr_mask).tocsr()
        bdry_mtx  = extract_cols_csc(mrows_mtx, bdry_mask).tocsr()
    else:
        intr_mtx = np.zeros([1])
        bdry_mtx = np.zeros([1])
    return [intr_mtx, bdry_mtx]

def split_matrix_mpi(mesh, M_MPI, global_intr_mask):
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    comm      = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()
    
    # Get idxs of intr, bdry DoFs
    global_intr_idxs = np.where(global_intr_mask)[0].astype(np.int32)
    global_bdry_idxs = np.where(np.invert(global_intr_mask))[0].astype(np.int32)
    
    # Broadcast to other ranks
    global_intr_idxs = MPI_comm.bcast(global_intr_idxs, root = 0)
    global_bdry_idxs = MPI_comm.bcast(global_bdry_idxs, root = 0)
    
    ii_0, ii_f = M_MPI.getOwnershipRange()
    local_intr_idxs  = global_intr_idxs[(global_intr_idxs >= ii_0)
                                        * (global_intr_idxs < ii_f)]
    intr_IS = PETSc.IS()
    intr_IS.createGeneral(local_intr_idxs, comm = comm)
    local_bdry_idxs  = global_bdry_idxs[(global_bdry_idxs >= ii_0)
                                        * (global_bdry_idxs < ii_f)]
    bdry_IS = PETSc.IS()
    bdry_IS.createGeneral(local_bdry_idxs, comm = comm)
    
    M_MPI_intr = M_MPI.createSubMatrix(isrow = intr_IS, iscol = intr_IS)
    M_MPI_bdry = M_MPI.createSubMatrix(isrow = intr_IS, iscol = bdry_IS)
    
    return [M_MPI_intr, M_MPI_bdry]

def split_vector(mesh, vec, intr_mask):
    return split_vector_mpi(mesh, vec, intr_mask)

def split_vector_mpi(mesh, v_MPI, global_intr_mask):
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    comm      = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()

    # Get idxs of intr, bdry DoFs
    global_intr_idxs = np.where(global_intr_mask)[0].astype(np.int32)
    global_bdry_idxs = np.where(np.invert(global_intr_mask))[0].astype(np.int32)

    # Broadcast to other ranks
    global_intr_idxs = MPI_comm.bcast(global_intr_idxs, root = 0)
    global_bdry_idxs = MPI_comm.bcast(global_bdry_idxs, root = 0)

    ii_0, ii_f       = v_MPI.getOwnershipRange()
    local_intr_idxs  = global_intr_idxs[(global_intr_idxs >= ii_0)
                                        * (global_intr_idxs < ii_f)]
    intr_IS = PETSc.IS()
    intr_IS.createGeneral(local_intr_idxs, comm = comm)
    local_bdry_idxs  = global_bdry_idxs[(global_bdry_idxs >= ii_0)
                                        * (global_bdry_idxs < ii_f)]
    bdry_IS = PETSc.IS()
    bdry_IS.createGeneral(local_bdry_idxs, comm = comm)
    
    v_MPI_intr = v_MPI.getSubVector(iset = intr_IS)
    v_MPI_bdry = v_MPI.getSubVector(iset = bdry_IS)
    
    return [v_MPI_intr, v_MPI_bdry]

def merge_vectors(intr_vec, bdry_vec, intr_mask):
    return merge_vectors_mpi(intr_vec, bdry_vec, intr_mask)

def merge_vectors_mpi(intr_vec, bdry_vec, intr_mask):
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    comm      = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()
    
    intr_ii_0, intr_ii_f = intr_vec.getOwnershipRange()
    global_intr_vec      = MPI_comm.gather(intr_vec[intr_ii_0:intr_ii_f], root = 0)
    
    bdry_ii_0, bdry_ii_f = bdry_vec.getOwnershipRange()
    global_bdry_vec      = MPI_comm.gather(bdry_vec[bdry_ii_0:bdry_ii_f], root = 0)
    
    if comm_rank == 0:
        global_intr_vec = np.concatenate(global_intr_vec)
        global_bdry_vec = np.concatenate(global_bdry_vec)
        
        ndof = np.size(intr_mask)
        
        vec = np.zeros(ndof)
        intr_idx = 0
        bdry_idx = 0
        
        for ii in range(0, ndof):
            if intr_mask[ii]:
                vec[ii] = global_intr_vec[intr_idx]
                
                intr_idx += 1
            else:
                vec[ii] = global_bdry_vec[bdry_idx]
                
                bdry_idx += 1
    else:
        vec = None
        
    return vec

def merge_vectors_seq(intr_vec, bdry_vec, intr_mask):
    
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
