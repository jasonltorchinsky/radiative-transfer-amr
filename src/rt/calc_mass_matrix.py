import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag, bmat, diags

from dg.matrix import get_idx_map, get_col_idxs, get_cell_idxs
from dg.projection import push_forward
import dg.quadrature as qd

def calc_mass_matrix(mesh, kappa):

    # Create column indexing for constructing global mass matrix
    [ncols, col_idxs] = get_col_idxs(mesh)
    col_mtxs = [None] * ncols # Global mass matrix is block-diagonal, and so
                              # there are only ncol non-zero column mass matrices

    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            # Get column information, quadrature weights
            col_idx = col_idxs[col_key]
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy] = [x1 - x0, y1 - y0]
            [ndof_x, ndof_y] = col.ndofs[:]
            
            [xxb, w_x, yyb, w_y, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                      nnodes_y = ndof_y)
            
            # We reshape array to leverage vectorized operations of numpy
            w_x = w_x.reshape(ndof_x, 1, 1)
            w_y = w_y.reshape(1, ndof_y, 1)
            
            xxf = push_forward(x0, x1, xxb).reshape(ndof_x, 1)
            yyf = push_forward(y0, y1, yyb).reshape(1, ndof_y)

            kappa_col = kappa(xxf, yyf).reshape(ndof_x, ndof_y, 1)
            wx_wy_kappa_col = w_x * w_y * kappa_col
            
            # Create cell indexing for constructing column mass matrix
            [ncells, cell_idxs] = get_cell_idxs(mesh, col_key)
            cell_mtxs = [None] * ncells # Column mass matrix is block-diagonal, and
                                        # so there are only ncell non-zero cell
                                        # mass matrices

            for cell_key, cell in sorted(col.cells.items()):
                if cell.is_lf:
                    # Get cell information, quadrature weights
                    cell_idx   = cell_idxs[cell_key]
                    [th0, th1] = cell.pos
                    dth        = th1 - th0
                    [ndof_th]  = cell.ndofs

                    [_, _, _, _, _, w_th] = qd.quad_xyth(nnodes_th = ndof_th)

                    w_th = w_th.reshape(1, 1, ndof_th)
                    
                    wx_wy_kappa_cell = np.tile(wx_wy_kappa_col, (1, 1, ndof_th))
                    
                    dcoeff = dx * dy * dth / 8
                    
                    diagonal = (dcoeff * w_th * wx_wy_kappa_cell).flatten()
                    cell_mtxs[cell_idx] = diags(diagonal)
                    
                    """
                    # List of coordinates, values for constructing cell matrices
                    # NOTE: alpha and beta indices are the same since cell
                    # mass matrices are diagonal, so we will only calculate the
                    # beta index
                    cell_ndof = ndof_x * ndof_y * ndof_th
                    betalist  = np.zeros([cell_ndof], dtype = np.int32) # beta index
                    vlist     = np.zeros([cell_ndof]) # Entry value

                    # Indexing from i, j, a to beta
                    # In this case, the alpha and beta indices are the same,
                    # so we don't have to do them separately
                    beta = get_idx_map(ndof_x, ndof_y, ndof_th)

                    # Values common to equation for each entry
                    dcoeff = dx * dy * dth / 8

                    # Construct cell matrix
                    idx = 0
                    for ii in range(0, ndof_x):
                        wx_i = w_x[ii]
                        for jj in range(0, ndof_y):
                            wy_j = w_y[jj]
                            kappa_ij = kappa(xxf[ii], yyf[jj])
                            for aa in range(0, ndof_th):
                                wth_a = w_th[aa]

                                # Calculate entry index, value
                                betalist[idx] = beta(ii, jj, aa)
                                
                                vlist[idx] = dcoeff * wx_i * wy_j * wth_a * kappa_ij
                                
                                idx += 1
                    
                    cell_mtxs[cell_idx] = coo_matrix((vlist, (betalist, betalist)))
                    """
                    
            col_mtxs[col_idx] = block_diag(cell_mtxs, format = 'csr')

    # Global mass matrix is block-diagonal
    # with the column matrices as the blocks
    mass_mtx = block_diag(col_mtxs, format = 'csr')

    return mass_mtx
