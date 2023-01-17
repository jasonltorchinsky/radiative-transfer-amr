import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag, bmat

from .Projection import Projection_2D

import dg.quadrature as qd
from dg.mesh import ji_mesh, tools

def calc_int_conv_matrix(mesh):

    # Store local-colmun matrices in here
    ncols = len(mesh.cols.keys())
    col_mtxs = ncols * [0]
    col_idx = 0

    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            # Get column information
            [x0, y0, x1, y1] = col.pos
            dx = x1 - x0
            dy = y1 - y0
            [dof_x, dof_y] = col.ndofs
            
            [nodes_x, weights_x, nodes_y, weights_y, _, _] = qd.quad_xya(dof_x, dof_y, 1)
            
            # Store local-element matrices in here
            ncells = len(list(col.cells.keys()))
            cell_mtxs = ncells * [0]
            cell_idx = 0

            for cell_key, cell in sorted(col.cells.items()):
                if cell.is_lf:
                    [dof_a] = cell.ndofs

                    # Indexing from i, j, a to beta
                    def beta(ii, jj, aa):
                        val = dof_a * dof_y * ii \
                            + dof_a * jj \
                            + aa
                        return val
                    
                    # Get cell information
                    cell_ndof = dof_x * dof_y * dof_a
                    [a0, a1] = cell.pos
                    da = a1 - a0
                    dcoeff = dx * dy * da / 8
                    
                    [_, _, _, _, nodes_a, weights_a] = qd.quad_xya(1, 1, dof_a)
                    th = norm_to_local(a0, a1, nodes_a)

                    # Construct delta_ip * delta_ar term
                    cell_mtx_ndof_ipar = dof_x * dof_y**2 * dof_a # Number of non-zero terms
                    
                    alphalist_ipar = np.zeros([cell_mtx_ndof_ipar],
                                         dtype = np.int32) # alpha index
                    betalist_ipar = np.zeros([cell_mtx_ndof_ipar],
                                        dtype = np.int32) # beta index
                    vlist_ipar = np.zeros([cell_mtx_ndof_ipar]) # Matrix entry
                    cnt = 0

                    for ii in range(0, dof_x):
                        wx_i = weights_x[ii]
                        for jj in range(0, dof_y):
                            wy_j = weights_y[jj]
                            for aa in range(0, dof_a):
                                wth_a = weights_a[aa]
                                for qq in range(0, dof_y):
                                    betalist_ipar[cnt] = beta(ii, qq, aa)
                                    alphalist_ipar[cnt] = beta(ii, jj, aa)
                                    # NEED TO FIX THIS FORMULA
                                    vlist_ipar[cnt] = dcoeff * wx_i * wy_j * wth_a \
                                        * qd.gl_deriv(nodes_y, qq, nodes_y[jj]) * np.sin(th[aa])
                                    cnt += 1

                    # Construct delta_jq * delta_ar term
                    cell_mtx_ndof_jqar = dof_x**2 * dof_y dof_a # Number of non-zero terms
                    
                    alphalist_jqar = np.zeros([cell_mtx_ndof_ipar],
                                         dtype = np.int32) # alpha index
                    betalist_jqar = np.zeros([cell_mtx_ndof_ipar],
                                        dtype = np.int32) # beta index
                    vlist_jqar = np.zeros([cell_mtx_ndof_ipar]) # Matrix entry
                    cnt = 0
                    
                    for ii in range(0, dof_x):
                        wx_i = weights_x[ii]
                        for jj in range(0, dof_y):
                            wy_j = weights_y[jj]
                            for aa in range(0, dof_a):
                                wth_a = weights_a[aa]
                                for pp in range(0, dof_x):
                                    betalist_jqar[cnt] = beta(pp, jj, aa)
                                    alphalist_jqar[cnt] = beta(ii, jj, aa)
                                    # NEED TO FIX THIS FORMULA
                                    vlist_jqar[cnt] = dcoeff * wx_i * wy_j * wth_a \
                                        * qd.gl_deriv(nodes_x, pp, nodes_x[ii]) * np.cos(th[aa])
                                    cnt += 1
                                
                    cell_mtxs[cell_idx] = coo_matrix((vlist_ipar, (alphalist_ipar, betalist_ipar))) + coo_matrix((vlist_jqar, (alphalist_jqar, betalist_jqar)))
                    cell_idx += 1
                    
            col_mtxs[col_idx] = block_diag(cell_mtxs, format = 'csr')
            
            col_idx += 1
                
    M_int_conv = block_diag(col_mtxs, format = 'csr')

    return M_int_conv
