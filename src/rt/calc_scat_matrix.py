import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag, bmat
from time import perf_counter

from dg.matrix import get_idx_map, get_col_idxs, get_cell_idxs
from dg.projection import push_forward
import dg.quadrature as qd

from utils import print_msg

def calc_scat_matrix(mesh, sigma, Phi, **kwargs):
    return calc_scat_matrix_old(mesh, sigma, Phi, **kwargs)

def calc_scat_matrix_new(mesh, sigma, Phi, **kwargs):
    
    default_kwargs = {'verbose' : False}
    kwargs = {**default_kwargs, **kwargs}
    
    if kwargs['verbose']:
        t0 = perf_counter()
    
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
            
            xxf = push_forward(x0, x1, xxb).reshape([nx, 1])
            wx  = wx.reshape([nx, 1])
            yyf = push_forward(y0, y1, yyb).reshape([1, ny])
            wy  = wy.reshape([1, ny])
            wx_wy_sigma_h = wx * wy * sigma(xxf, yyf)
            
            
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

                            nth_d = 15
                            [_, _, _, _, thb_1, wth_1] = qd.quad_xyth(nnodes_th = nth_1)
                            [_, _, _, _, thb_d, wth_d] = qd.quad_xyth(nnodes_th = nth_d)
                            thf_d = push_forward(th0_1, th1_1, thb_d)
                            
                            Phi_h = np.zeros([nth_0, nth_d])
                            for rr in range(0, nth_0):
                                for aa_p in range(0, nth_d):
                                    Phi_h[rr, aa_p] = Phi(thf_0[rr], thf_d[aa_p])
                            
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
                                for jj in range(0, ny):
                                    wx_wy_sigma_ij = wx_wy_sigma_h[ii, jj]
                                    if np.abs(wx_wy_sigma_ij) > 1.e-14:
                                        for rr in range(0, nth_0):
                                            wth_0_rr = wth_0[rr]
                                            for aa in range(0, nth_1):
                                                val = 0
                                                for aa_p in range(0, nth_d):
                                                    wth_d_aap = wth_d[aa_p]
                                                    Phi_rap = Phi_h[rr, aa_p]
                                                    
                                                    val += wth_d_aap * Phi_rap * qd.lag_eval(thb_d, aa_p, thb_1[aa])
                                                    
                                                val *= dcoeff * (dth_1 / 2.0) \
                                                    * wx_wy_sigma_ij * wth_0_rr
                                                
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
                            
                            # Clean up some variables
                            del alphalist, betalist, vlist, Phi_h
                            
            # Column scattering matrix is not block-diagonal
            # but we arranged the cell matrices in the proper form
            col_mtxs[col_idx] = bmat(cell_mtxs, format = 'coo')

            # Clean up some variables
            del cell_mtxs

    # Global scattering matrix is block-diagonal
    # with the column matrices as the blocks
    scat_mtx = block_diag(col_mtxs, format = 'csr')

    # Clean up some variables
    del col_mtxs
    
    if kwargs['verbose']:
        tf = perf_counter()
        msg = (
            'Scattering Matrix Construction Time: {:8.4f} [s]\n'.format(tf - t0)
            )
        print_msg(msg)

    return scat_mtx

def calc_scat_matrix_old(mesh, sigma, Phi, **kwargs):
    
    default_kwargs = {'verbose' : False}
    kwargs = {**default_kwargs, **kwargs}
    
    if kwargs['verbose']:
        t0 = perf_counter()
    
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
            
            xxf = push_forward(x0, x1, xxb).reshape([nx, 1])
            wx  = wx.reshape([nx, 1])
            yyf = push_forward(y0, y1, yyb).reshape([1, ny])
            wy  = wy.reshape([1, ny])
            wx_wy_sigma_h = wx * wy * sigma(xxf, yyf)
            
            
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
                                for jj in range(0, ny):
                                    wx_wy_sigma_ij = wx_wy_sigma_h[ii, jj]
                                    if np.abs(wx_wy_sigma_ij) > 1.e-14:
                                        for rr in range(0, nth_0):
                                            wth_0_rr = wth_0[rr]
                                            for aa in range(0, nth_1):
                                                wth_1_aa = wth_1[aa]
                                                Phi_ra = Phi_h[rr, aa]
                                                
                                                val = dcoeff * (dth_1 / 2.0) \
                                                    * wx_wy_sigma_ij * wth_0_rr \
                                                    * wth_1_aa * Phi_ra
                                                
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
                            
                            # Clean up some variables
                            del alphalist, betalist, vlist, Phi_h
                            
            # Column scattering matrix is not block-diagonal
            # but we arranged the cell matrices in the proper form
            col_mtxs[col_idx] = bmat(cell_mtxs, format = 'coo')

            # Clean up some variables
            del cell_mtxs

    # Global scattering matrix is block-diagonal
    # with the column matrices as the blocks
    scat_mtx = block_diag(col_mtxs, format = 'csr')

    # Clean up some variables
    del col_mtxs
    
    if kwargs['verbose']:
        tf = perf_counter()
        msg = (
            'Scattering Matrix Construction Time: {:8.4f} [s]\n'.format(tf - t0)
            )
        print_msg(msg)

    return scat_mtx

