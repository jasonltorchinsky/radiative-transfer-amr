import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag, bmat

from dg.matrix import get_idx_map, get_col_idxs, get_cell_idxs
from dg.projection import push_forward
import dg.quadrature as qd

def calc_scat_matrix(mesh, sigma, Phi):
    # Create column indexing for constructing global mass matrix
    [ncols, col_idxs] = get_col_idxs(mesh)
    col_mtxs = [None] * ncols # Global scattering matrix is block-diagonal, and so
                              # there are only ncol non-zero column scattering matrices
    
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            # Get column information, quadrature weights
            col_idx = col_idxs[col_key]
            [x0, y0, x1, y1] = col.pos
            [dx, dy] = [x1 - x0, y1 - y0]
            [nx, ny] = col.ndofs
            
            [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = nx,
                                                    nnodes_y = ny)
            
            xxf = push_forward(x0, x1, xxb)
            yyf = push_forward(y0, y1, yyb)
            
            sigma_h = np.zeros([nx, ny])
            for ii in range(0, nx):
                for jj in range(0, ny):
                    sigma_h[ii, jj] = sigma(xxf[ii], yyf[jj])
            
            # Create cell indexing for constructing column mass matrix
            [ncells, cell_idxs] = get_cell_idxs(mesh, col_key)
            cell_mtxs = [[None] * ncells for K in range(0, ncells)]
                            # Column matrix is block-dense, and so there are
                            # up to ncell**2 non-zero cell scattering matrices
                            
            # _0 refers to element K in the equations
            # _1 refers to element K' in the equations
            for cell_key_0, cell_0 in sorted(col.cells.items()):
                if cell_0.is_lf:
                    # Get cell information, quadrature weights
                    cell_idx_0     = cell_idxs[cell_key_0]
                    [th0_0, th1_0] = cell_0.pos
                    dth_0          = th1_0 - th0_0
                    [nth_0]        = cell_0.ndofs
                    
                    [_, _, _, _, thb_0, wth_0] = qd.quad_xyth(nnodes_th = nth_0)
                    
                    thf_0 = push_forward(th0_0, th1_0, thb_0)
                    
                    # Indexing from p, q, r to alpha
                    alpha = get_idx_map(nx, ny, nth_0)
                    
                    # Values common to equation for each entry
                    dcoeff = dx * dy * dth_0 / 8
                    
                    for cell_key_1, cell_1 in sorted(col.cells.items()):
                        if cell_1.is_lf:
                            # Get cell information, quadrature weights
                            cell_idx_1     = cell_idxs[cell_key_1]
                            [th0_1, th1_1] = cell_1.pos
                            dth_1          = th1_1 - th0_1
                            [nth_1]        = cell_1.ndofs
                            
                            [_, _, _, _, thb_1, wth_1] = qd.quad_xyth(nnodes_th = nth_1)
                            thf_1 = push_forward(th0_1, th1_1, thb_1)
                            
                            Phi_h = np.zeros([nth_0, nth_1])
                            for rr in range(0, nth_0):
                                for aa in range(0, nth_1):
                                    Phi_h[rr, aa] = Phi(thf_0[rr], thf_1[aa])
                            
                            # List of coordinates, values for constructing cell matrices
                            cell_ndof = nth_0 * nth_1 * nx * ny
                            alphalist = np.zeros([cell_ndof], dtype = np.int32) # alpha index
                            betalist  = np.zeros([cell_ndof], dtype = np.int32) # beta index
                            vlist     = np.zeros([cell_ndof]) # Matrix entry
                            
                            # Indexing from i, j, a to beta
                            beta = get_idx_map(nx, ny, nth_1)
                            
                            # Construct cell matrix
                            idx = 0
                            for ii in range(0, nx):
                                wx_i = wx[ii]
                                for jj in range(0, ny):
                                    wy_j = wy[jj]
                                    sigma_ij = sigma_h[ii, jj]
                                    for rr in range(0, nth_0):
                                        wth_0_rr = wth_0[rr]
                                        for aa in range(0, nth_1):
                                            wth_1_aa = wth_1[aa]
                                            Phi_ra = Phi_h[rr, aa]
                                            
                                            val = dcoeff * (dth_1 / 2.0) \
                                                * wx_i * wy_j * wth_0_rr \
                                                * wth_1_aa * sigma_ij * Phi_ra
                                            
                                            if np.abs(val) > 1.e-14:
                                                # Index of entry
                                                alphalist[idx] = alpha(ii, jj, rr)
                                                betalist[idx]  = beta( ii, jj, aa)
                                                
                                                vlist[idx] = val
                                                idx += 1
                                        
                            cell_mtxs[cell_idx_0][cell_idx_1] = coo_matrix((vlist, (alphalist, betalist)),
                                                                           shape = (nx * ny * nth_0,
                                                                                    nx * ny * nth_1))
                            cell_mtxs[cell_idx_0][cell_idx_1].eliminate_zeros()
                            
            # Column scattering matrix is not block-diagonal
            # but we arranged the cell matrices in the proper form
            col_mtxs[col_idx] = bmat(cell_mtxs, format = 'csr')

    # Global scattering matrix is block-diagonal
    # with the column matrices as the blocks
    scat_mtx = block_diag(col_mtxs, format = 'csr')

    return scat_mtx


def calc_scat_matrix_old(mesh, sigma, Phi):
    # Create column indexing for constructing global mass matrix
    [ncols, col_idxs] = get_col_idxs(mesh)
    col_mtxs = [None] * ncols # Global scattering matrix is block-diagonal, and so
                              # there are only ncol non-zero column scattering matrices
    
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            # Get column information, quadrature weights
            col_idx = col_idxs[col_key]
            [x0, y0, x1, y1] = col.pos
            [dx, dy] = [x1 - x0, y1 - y0]
            [ndof_x, ndof_y] = col.ndofs
            
            [xxb, w_x, yyb, w_y, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                      nnodes_y = ndof_y)
            
            xxf = push_forward(x0, x1, xxb).reshape(ndof_x, 1)
            yyf = push_forward(y0, y1, yyb).reshape(1, ndof_y)

            w_x = w_x.reshape(ndof_x, 1)
            w_y = w_y.reshape(1, ndof_y)

            wx_wy_sigma_col = w_x * w_y * sigma(xxf, yyf)
            
            # Create cell indexing for constructing column mass matrix
            [ncells, cell_idxs] = get_cell_idxs(mesh, col_key)
            cell_mtxs = [[None] * ncells for K in range(0, ncells)]
                            # Column matrix is block-dense, and so there are
                            # up to ncell**2 non-zero cell scattering matrices
                            
            # _0 refers to element K in the equations
            # _1 refers to element K' in the equations
            for cell_key_0, cell_0 in sorted(col.cells.items()):
                if cell_0.is_lf:
                    # Get cell information, quadrature weights
                    cell_idx_0     = cell_idxs[cell_key_0]
                    [th0_0, th1_0] = cell_0.pos
                    dth_0          = th1_0 - th0_0
                    [ndof_th_0]    = cell_0.ndofs
                    
                    [_, _, _, _, thb_0, w_th_0] = qd.quad_xyth(nnodes_th = ndof_th_0)
                    thf_0 = push_forward(th0_0, th1_0, thb_0).reshape(ndof_th_0, 1)
                    w_th_0 = w_th_0.reshape(ndof_th_0, 1)
                    
                    # Indexing from p, q, r to alpha
                    alpha = get_idx_map(ndof_x, ndof_y, ndof_th_0)
                    
                    # Values common to equation for each entry
                    dcoeff = dx * dy * dth_0 / 8
                    
                    for cell_key_1, cell_1 in sorted(col.cells.items()):
                        if cell_1.is_lf:
                            # Get cell information, quadrature weights
                            cell_idx_1     = cell_idxs[cell_key_1]
                            [th0_1, th1_1] = cell_1.pos
                            dth_1          = th1_1 - th0_1
                            [ndof_th_1]    = cell_1.ndofs
                            
                            [_, _, _, _, thb_1, w_th_1] = qd.quad_xyth(nnodes_th = ndof_th_1)
                            thf_1 = push_forward(th0_1, th1_1, thb_1).reshape(1, ndof_th_1)
                            w_th_1 = w_th_1.reshape(1, ndof_th_1)
                            
                            Phi_cell = Phi(thf_0, thf_1)
                            
                            wth0_wth1_Phi_cell = w_th_0 * w_th_1 * Phi_cell
                            
                            # List of coordinates, values for constructing cell matrices
                            cell_ndof = ndof_th_0 * ndof_th_1 * ndof_x * ndof_y
                            alphalist = np.zeros([cell_ndof], dtype = np.int32) # alpha index
                            betalist  = np.zeros([cell_ndof], dtype = np.int32) # beta index
                            vlist = np.zeros([cell_ndof]) # Matrix entry
                            
                            # Indexing from i, j, a to beta
                            beta = get_idx_map(ndof_x, ndof_y, ndof_th_1)
                            
                            # Construct cell matrix
                            idx = 0
                            for ii in range(0, ndof_x):
                                for jj in range(0, ndof_y):
                                    wx_wy_sigma_ij = wx_wy_sigma_col[ii, jj]
                                    for rr in range(0, ndof_th_0):
                                        for aa in range(0, ndof_th_1):                 
                                            wth0_wth1_Phi_ra = wth0_wth1_Phi_cell[rr, aa]

                                            val = dcoeff * (dth_1 / 2.0) \
                                                * wx_wy_sigma_ij * wth0_wth1_Phi_ra

                                            if np.abs(val) > 1.e-14:
                                                # Index of entry
                                                alphalist[idx] = alpha(ii, jj, rr)
                                                betalist[idx]  = beta( ii, jj, aa)
                                                
                                                vlist[idx] = dcoeff * (dth_1 / 2.0) \
                                                    * wx_wy_sigma_ij * wth0_wth1_Phi_ra
                                                idx += 1
                                        
                            cell_mtxs[cell_idx_0][cell_idx_1] = coo_matrix((vlist, (alphalist, betalist)),
                                                                           shape = (ndof_x * ndof_y * ndof_th_0,
                                                                                    ndof_x * ndof_y * ndof_th_1))
                            cell_mtxs[cell_idx_0][cell_idx_1].eliminate_zeros()
                            
            # Column scattering matrix is not block-diagonal
            # but we arranged the cell matrices in the proper form
            col_mtxs[col_idx] = bmat(cell_mtxs, format = 'csr')

    # Global scattering matrix is block-diagonal
    # with the column matrices as the blocks
    scat_mtx = block_diag(col_mtxs, format = 'csr')

    return scat_mtx
