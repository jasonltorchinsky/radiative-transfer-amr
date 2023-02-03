import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix

from .Projection import Projection_2D

import dg.quadrature as qd
from dg.mesh import ji_mesh, tools

def push_forward(x0, xf, nodes):
    """
    Transforms nodes on [-1, 1] to a [x0, xf].
    """

    xx = x0 + (xf - x0) / 2.0 * (nodes + 1.0)

    return xx

def pull_back(x0, xf, nodes):
    """
    Transforms nodes on [x0, xf] to a [-1, 1].
    """

    xx = -1.0 +  2.0 / (xf - x0) * (nodes - x0)

    return xx

def get_bdry_idxs(mesh):

    mesh_ndof = 0
    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            [ndof_x, ndof_y] = col.ndofs
            for cell_key, cell in sorted(col.cells.items()):
                if cell.is_lf:
                    [ndof_th] = cell.ndofs

                    mesh_ndof += ndof_x * ndof_y * ndof_th

    return mesh_ndof
                    

def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype = bool)
    mask[indices] = False
    return mat[mask]

def delete_cols_csc(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, csc_matrix):
        raise ValueError("works only for CSC format -- use .tocsc() first")
    indices = list(indices)
    mask = np.ones(mat.shape[1], dtype = bool)
    mask[indices] = False
    return mat[:, mask]

def split_mtx(mat, intr_idxs, bdry_idxs):

    mtx = mat.tocsr()
    mrows_mtx = delete_rows_csr(mtx, bdry_idxs)
    intr_mtx  = delete_cols_csc(mrows_mtx.tocsc(), bdry_idxs)
    bdry_mtx  = delete_cols_csc(mrows_mtx.tocsc(), intr_idxs)

    return [intr_mtx, bdry_mtx]
