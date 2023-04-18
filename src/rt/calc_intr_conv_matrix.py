import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag, bmat

from dg.matrix import get_idx_map, get_col_idxs, get_cell_idxs
from dg.projection import push_forward
import dg.quadrature as qd

def calc_intr_conv_matrix(mesh):

    # Create column indexing for constructing global interior convection  matrix
    [ncols, col_idxs] = get_col_idxs(mesh)
    col_mtxs = [None] * ncols # Global interior convection matrix is block-diagonal,
                              # and so there are only ncol non-zero column mass matrices

    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            # Get column information, quadrature weights
            col_idx = col_idxs[col_key]
            [x0, y0, x1, y1] = col.pos
            [dx, dy] = [x1 - x0, y1 - y0]
            [ndof_x, ndof_y] = col.ndofs
            
            [xxb, w_x, yyb, w_y, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                      nnodes_y = ndof_y)

            w_x = w_x.reshape([ndof_x, 1])
            w_y = w_y.reshape([1, ndof_y])

            wx_wy = w_x * w_y
            
            # Create cell indexing for constructing column mass matrix
            [ncells, cell_idxs] = get_cell_idxs(mesh, col_key)
            cell_mtxs = [None] * ncells # Column interior convection  matrix is
                                        # block-diagonal, and so there are only 
                                        # ncell non-zero cell interior convection
                                        # matrices

            for cell_key, cell in sorted(col.cells.items()):
                if cell.is_lf:
                    # Get cell information, quadrature weights
                    cell_idx   = cell_idxs[cell_key]
                    [th0, th1] = cell.pos
                    dth        = th1 - th0
                    [ndof_th]  = cell.ndofs
                    
                    [_, _, _, _, thb, w_th] = qd.quad_xyth(nnodes_th = ndof_th)
                    
                    thf = push_forward(th0, th1, thb)

                    wth_sinth = w_th * np.sin(thf)
                    wth_costh = w_th * np.cos(thf)

                    # Indexing from i, j, a to beta
                    # Same formula for p, q, r to alpha, but we define alpha
                    # anyway for clarity
                    alpha = get_idx_map(ndof_x, ndof_y, ndof_th)
                    beta  = get_idx_map(ndof_x, ndof_y, ndof_th)

                    # Set up arrays for delta_ip * delta_ar term
                    cell_ndof      = ndof_x * ndof_y * ndof_th
                    cell_ndof_ipar = ndof_x * ndof_y**2 * ndof_th
                    alphalist_ipar = np.zeros([cell_ndof_ipar], dtype = np.int32)
                    betalist_ipar  = np.zeros([cell_ndof_ipar], dtype = np.int32)
                    vlist_ipar     = np.zeros([cell_ndof_ipar])
                    
                    # Construct delta_ip * delta_ar term
                    # i = p, a = r
                    # When we take the derivative of psi, we end up with
                    # 2/dy * psi_bar in normalized coordinates, so dcoeff is
                    dcoeff = dx * dth / 4.
                    idx = 0
                    for ii in range(0, ndof_x):
                        for jj in range(0, ndof_y):
                            wxi_wyj = wx_wy[ii, jj]
                            for aa in range(0, ndof_th):
                                wth_sinth_a = wth_sinth[aa]
                                for qq in range(0, ndof_y):
                                    ddy_psi_qj = qd.lag_ddx_eval(yyb, qq, yyb[jj])

                                    alphalist_ipar[idx] = alpha(ii, qq, aa)
                                    betalist_ipar[idx]  = beta( ii, jj, aa)
                                    
                                    vlist_ipar[idx] = dcoeff * wxi_wyj * wth_sinth_a \
                                        * ddy_psi_qj
                                    
                                    idx += 1
                                    
                    delta_ipar = coo_matrix((vlist_ipar,
                                             (alphalist_ipar, betalist_ipar)),
                                            shape = (cell_ndof, cell_ndof))
                    
                    
                    # Set up arrays for  delta_jq * delta_ar term
                    cell_ndof_jqar = ndof_x**2 * ndof_y * ndof_th                    
                    alphalist_jqar = np.zeros([cell_ndof_jqar], dtype = np.int32)
                    betalist_jqar  = np.zeros([cell_ndof_jqar], dtype = np.int32)
                    vlist_jqar     = np.zeros([cell_ndof_jqar])
                    
                    # Construct delta_jq * delta_ar term
                    # j = q, a = r
                    # When we take the derivative of phi, we end up with
                    # 2/dx * phi_bar in normalized coordinates, so dcoeff is
                    dcoeff = dy * dth / 4.
                    idx = 0
                    for ii in range(0, ndof_x):
                        for jj in range(0, ndof_y):
                            wxi_wyj = wx_wy[ii, jj]
                            for aa in range(0, ndof_th):
                                wth_costh_a = wth_costh[aa]
                                for pp in range(0, ndof_x):
                                    ddx_phi_pi = qd.lag_ddx_eval(xxb, pp, xxb[ii])
                                    
                                    alphalist_jqar[idx] = alpha(pp, jj, aa)
                                    betalist_jqar[idx]  = beta( ii, jj, aa)
                                    
                                    vlist_jqar[idx] = dcoeff * wxi_wyj * wth_costh_a \
                                        * ddx_phi_pi
                                    
                                    idx += 1
                    
                    delta_jqar = coo_matrix((vlist_jqar,
                                             (alphalist_jqar, betalist_jqar)),
                                            shape = (cell_ndof, cell_ndof))
                    
                    cell_mtxs[cell_idx] = delta_ipar + delta_jqar
                    
            col_mtxs[col_idx] = block_diag(cell_mtxs, format = 'csr')
            
    # Global interior convection matrix is block-diagonal
    # with the column matrices as the blocks
    intr_conv_mtx = block_diag(col_mtxs, format = 'csr')

    return intr_conv_mtx
