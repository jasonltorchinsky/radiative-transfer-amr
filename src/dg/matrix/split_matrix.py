import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

from .get_masks import get_intr_mask, get_bdry_mask

def extract_rows_csr(mat, mask):
    """
    Remove the rows denoted by ``indices`` from the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    return mat[mask]

def extract_cols_csc(mat, mask):
    """
    Remove the columns denoted by ``indices`` from the CSC sparse matrix ``mat``.
    """
    if not isinstance(mat, csc_matrix):
        raise ValueError("works only for CSC format -- use .tocsc() first")
    return mat[:, mask]

def split_matrix(mesh, mat):

    intr_mask = get_intr_mask(mesh)
    bdry_mask = np.invert(intr_mask)
    
    mtx = mat.tocsr()
    
    mrows_mtx = extract_rows_csr(mtx, intr_mask)
    mrows_mtx = mrows_mtx.tocsc()
    
    intr_mtx  = extract_cols_csc(mrows_mtx, intr_mask)
    bdry_mtx  = extract_cols_csc(mrows_mtx, bdry_mask)

    return [intr_mtx, bdry_mtx]
