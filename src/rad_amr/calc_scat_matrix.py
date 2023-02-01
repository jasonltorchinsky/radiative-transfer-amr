import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag, bmat

from .Projection import Projection_2D

import dg.quadrature as qd
from .matrix_utils import push_forward, pull_back

def calc_scat_matrix(mesh, sigma, Phi):

    sigmah = Projection_2D(mesh, sigma)

    # Create column indexing for constructing global mass matrix
    col_idx = 0
    col_idxs = dict()
    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            col_idxs[col_key] = col_idx
            col_idx += 1

    ncols = col_idx # col_idx counts the number of existing columns in mesh
    col_mtxs = [None] * ncols # Global scattering matrix is block-diagonal, and so
                              # there are only ncol non-zero column scattering matrices

    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            # Get column information, quadrature weights
            col_idx = col_idxs[col_key]
            [x0, y0, x1, y1] = col.pos
            dx = x1 - x0
            dy = y1 - y0
            [ndof_x, ndof_y] = col.ndofs
            
            [_, w_x, _, w_y, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                  nnodes_y = ndof_y)
            
            # Get values of sigma in column
            sigmah_col = sigmah.cols[col_key].vals
            
            # Create cell indexing for constructing column mass matrix
            cell_idx = 0
            cell_idxs = dict()
            for cell_key, cell in sorted(col.cells.items()):
                if cell.is_lf:
                    cell_idxs[cell_key] = cell_idx
                    cell_idx += 1
                    
            ncells = cell_idx # cell_idx counts the number of existing cells in column
            cell_mtxs = [[None] * ncells for K in range(0, ncells)]
                            # Column matrix is block-dense, and so there are
                            # up to ncell**2 non-zero cell scattering matrices
                            
            # _0 refers to element K' in the equations
            # _1 refers to element K in the equations
            for cell_key_0, cell_0 in sorted(col.cells.items()):
                if cell_0.is_lf:
                    # Get cell information, quadrature weights
                    cell_idx_0     = cell_idxs[cell_key_0]
                    [th0_0, th1_0] = cell_0.pos
                    dth_0          = th1_0 - th0_0
                    [ndof_th_0]    = cell_0.ndofs
                    
                    [_, _, _, _, thb_0, w_th_0] = qd.quad_xyth(nnodes_th = ndof_th_0)
                    thf_0 = push_forward(th0_0, th1_0, thb_0)
                    
                    # Indexing from p, q, r to alpha
                    def alpha(pp, qq, rr):
                        val = ndof_th_0 * ndof_y * pp \
                            + ndof_th_0 * qq \
                            + rr
                        return val
                    
                    # Values common to equation for each entry
                    dcoeff = dx * dy * dth_0 / 8
                    
                    for cell_key_1, cell_1 in sorted(col.cells.items()):
                        if cell_1.is_lf:
                            # Get cell information, quadrature weights
                            cell_idx_1     = cell_idxs[cell_key_1]
                            [th0_1, th1_1] = cell_1.pos
                            dth_1          = th1_1 - th0_1
                            [ndof_th_1]    = cell_1.ndofs
                            
                            [_, _, _, _, thb_1, w_th_1] = qd.quad_xyth(1, 1, ndof_th_1)
                            thf_1 = push_forward(th0_1, th1_1, thb_1)
                            
                            # List of coordinates, values for constructing cell matrices
                            cell_ndof = ndof_th_0 * ndof_th_1 * ndof_x * ndof_y
                            alphalist = np.zeros([cell_ndof], dtype = np.int32) # alpha index
                            betalist  = np.zeros([cell_ndof], dtype = np.int32) # beta index
                            vlist = np.zeros([cell_ndof]) # Matrix entry
                            
                            # Indexing from i, j, a to beta
                            def beta(ii, jj, aa):
                                val = ndof_th_1 * ndof_y * ii \
                                    + ndof_th_1 * jj \
                                    + aa
                                return val
                            
                            # Construct cell matrix
                            idx = 0
                            for ii in range(0, ndof_x):
                                wx_i = w_x[ii]
                                for jj in range(0, ndof_y):
                                    wy_j = w_y[jj]
                                    sigma_ij = sigmah_col[ii, jj]
                                    for rr in range(0, ndof_th_0):
                                        wth_r_0 = w_th_0[rr]
                                        for aa in range(0, ndof_th_1):
                                            wth_a_1 = w_th_1[aa]
                                            
                                            Phi_ar = Phi(thf_0[rr], thf_1[aa])
                                            
                                            # Index of entry
                                            alphalist[idx] = alpha(ii, jj, aa)
                                            betalist[idx]  = beta( ii, jj, rr)
                                            
                                            vlist[idx] = dcoeff * (dth_1 / 2.0) \
                                                * wx_i * wy_j * wth_r_0 * wth_a_1 \
                                                * sigma_ij * Phi_ar
                                            idx += 1
                                        
                            cell_mtxs[cell_idx_0][cell_idx_1] = coo_matrix((vlist, (alphalist, betalist)))

            # Column scattering matrix is not block-diagonal
            # but we arranged the cell matrices in the proper form
            col_mtxs[col_idx] = bmat(cell_mtxs, format = 'csr')

    # Global scattering matrix is block-diagonal
    # with the column matrices as the blocks
    scat_mtx = block_diag(col_mtxs, format = 'csr')

    return scat_mtx
