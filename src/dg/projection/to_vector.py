import numpy as np

from ..matrix import get_idx_map, get_col_idxs, get_cell_idxs

def to_vector(self):
    """
    Convert a projection to a vector.
    """

    [ncol, col_idxs] = get_col_idxs(self)
    col_vecs = [None] * ncol
    col_items = sorted(self.cols.items())

    for col_key, col in col_items:
        col_idx = col_idxs[col_key]
        [ndof_x, ndof_y] = col.ndofs[:]

        [ncell, cell_idxs] = get_cell_idxs(self, col_key)
        cell_vecs = [None] * ncell
        cell_items = sorted(col.cells.items())

        for cell_key, cell in cell_items:
            cell_idx = cell_idxs[cell_key]
            [ndof_th] = cell.ndofs

            beta = get_idx_map(ndof_x, ndof_y, ndof_th)

            cell_ndof = ndof_x * ndof_y * ndof_th
            cell_vec = np.zeros([cell_ndof])
            for ii in range(0, ndof_x):
                for jj in range(0, ndof_y):
                    for aa in range(0, ndof_th):
                        beta_idx = beta(ii, jj, aa)
                        cell_vec[beta_idx] = cell.vals[ii, jj, aa]

            cell_vecs[cell_idx] = cell_vec[:]

        col_vecs[col_idx] = np.concatenate(cell_vecs, axis = None)

    vec = np.concatenate(col_vecs, axis = None)

    return vec

