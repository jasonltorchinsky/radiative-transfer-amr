import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag, bmat

from .Projection import Projection_2D

import dg.quadrature as qd
from dg.mesh import ji_mesh, tools

def calc_mass_matrix(mesh, kappa):

    kappah = Projection_2D(mesh, kappa)

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
            
            [_, weights_x, _, weights_y, _, _] = qd.quad_xya(dof_x, dof_y, 1)
            
            kappah_col = kappah.cols[col_key].vals
            
            # Store local-element matrices in here
            ncells = len(list(col.cells.keys()))
            cell_mtxs = ncells * [0]
            cell_idx = 0

            for cell_key, cell in sorted(col.cells.items()):
                if cell.is_lf:
                    [dof_a] = cell.ndofs

                    # Indexing from i, j, a to beta
                    # In this case, the alpha and beta indices are the same,
                    # so we don't have to do them separately
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
                    
                    [_, _, _, _, _, weights_a] = qd.quad_xya(1, 1, dof_a)

                    # Lists for constructing diagonal matrices
                    cell_mtx_ndof = dof_x * dof_y * dof_a
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
                            kappa_ij = kappah_col[ii, jj]
                            for aa in range(0, dof_a):
                                wth_a = weights_a[aa]

                                # specify that the entry is on
                                betalist[cnt] = beta(ii, jj, aa)
                                alphalist[cnt] = betalist[cnt]
                                
                                vlist[cnt] = dcoeff * wx_i * wy_j * wth_a * kappa_ij
                                cnt += 1
                                
                    cell_mtxs[cell_idx] = coo_matrix((vlist, (alphalist, betalist)))
                    cell_idx += 1
                    
            col_mtxs[col_idx] = block_diag(cell_mtxs, format = 'csr')
            
            col_idx += 1
                
    M_mass = block_diag(col_mtxs, format = 'csr')

    return M_mass
