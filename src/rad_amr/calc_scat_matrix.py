import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag, bmat

from .Projection import Projection_2D

import dg.quadrature as qd
from dg.mesh import ji_mesh, tools

def calc_scat_matrix(mesh, sigma, phi):

    sigmah = Projection_2D(mesh, sigma)

    # Store local-colmun matrices in here
    ncols = len(mesh.cols.keys())
    col_mtxs = ncols * [0]
    col_idx = 0 # I don't think I actually increment this, which is certainly a bug!

    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            # Get column information
            [x0, y0, x1, y1] = col.pos
            dx = x1 - x0
            dy = y1 - y0
            [dof_x, dof_y] = col.ndofs
            
            [_, weights_x, _, weights_y, _, _] = qd.quad_xya(dof_x, dof_y, 1)
            
            sigmah_col = sigmah.cols[col_key].vals
            
            # Store local-element matrices in here
            ncells = len(list(col.cells.keys()))
            cell_mtxs = [[None] * ncells for K in range(0, ncells)]

            # _0 refers to element K' in the equations
            # _1 refers to element K in the equations
            cell_idx_0 = 0
            for cell_key_0, cell_0 in sorted(col.cells.items()):
                if cell_0.is_lf:
                    [dof_a_0] = cell_0.ndofs

                    # Indexing from i, j, a to beta
                    def alpha(pp, qq, rr):
                        val = dof_a_0 * dof_y * pp \
                            + dof_a_0 * qq \
                            + rr
                        return val
                    
                    # Get cell information
                    cell_0_ndof = dof_x * dof_y * dof_a_0
                    [a0_0, a1_0] = cell_0.pos
                    da_0 = a1_0 - a0_0
                    dcoeff = dx * dy * da_0 / 8
                    
                    [_, _, _, _, nodes_a_0, weights_a_0] = qd.quad_xya(1, 1, dof_a_0)
                    th_0 = norm_to_local(a0_0, a1_0, nodes_a_0)

                    cell_idx_1 = 0
                    for cell_key_1, cell_1 in sorted(col.cells.items()):
                        if cell_1.is_lf:
                            [dof_a_1] = cell_1.ndofs

                            # Indexing from i, j, a to beta
                            def beta(ii, jj, aa):
                                val = dof_a_1 * dof_y * ii \
                                    + dof_a_1 * jj \
                                    + aa
                                return val

                            # Get cell information
                            cell_1_ndof = dof_x * dof_y * dof_a_1
                            [a0_1, a1_1] = cell_1.pos
                            da_1 = a1_1 - a0_1

                            [_, _, _, _, nodes_a_1, weights_a_1] = qd.quad_xya(1, 1, dof_a_1)
                            th_1 = norm_to_local(a0_1, a1_1, nodes_a_1)

                            # Lists for constructing diagonal matrices
                            cell_mtx_ndof = dof_a_0 * dof_a_1 \
                                * dof_x * dof_y
                            alphalist = np.zeros([cell_mtx_ndof],
                                                 dtype = np.int32) # alpha index
                            betalist = np.zeros([cell_mtx_ndof],
                                                dtype = np.int32) # beta index
                            vlist = np.zeros([cell_mtx_ndof]) # Matrix entry
                            cnt = 0
                            
                            for ii in range(0, dof_x):
                                wx_i = weights_x[ii]
                                for jj in range(0, dof_y):
                                    wy_j = weights_y[jj]
                                    sigma_ij = sigmah_col[ii, jj]
                                    for rr in range(0, dof_a_0):
                                        wth_a_0 = weights_a_0[rr]
                                        for aa in range(0, dof_a_1):
                                            wth_a_1 = weights_a_1[aa]

                                            phi_ar = phi(th_0[rr], th_1[aa])

                                            # Index of entry
                                            alphalist[cnt] = alpha(ii, jj, aa)
                                            betalist[cnt] = beta(ii, jj, rr)
                                            
                                            vlist[cnt] = dcoeff * (da_1 / 2.0) * wx_i * wy_j * wth_a_0 * wth_a_1 * sigma_ij * phi_ar
                                            cnt += 1
                                        
                            cell_mtxs[cell_idx_0][cell_idx_1] = coo_matrix((vlist, (alphalist, betalist)))
                            cell_idx_1 += 1
                    cell_idx_0 += 1
                    
            col_mtxs[col_idx] = bmat(cell_mtxs, format = 'csr')
            
            col_idx += 1
                
    M_scat = block_diag(col_mtxs, format = 'csr')

    return M_scat
