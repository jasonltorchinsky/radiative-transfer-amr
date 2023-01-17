import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag, bmat

from .Projection import Projection_2D
from .matrix_utils import norm_to_local, local_to_norm, get_col_info, get_cell_info

import dg.quadrature as qd
from dg.mesh import ji_mesh, tools

def calc_bdry_conv_matrix(mesh):

    # Variables that are the same throughout the loops
    nhbr_locs = ['+', '-']
    def Theta(F, theta):
        if F == 0:
            return np.cos(theta)
        elif F == 1:
            return np.sin(theta)
        elif F == 2:
            return -np.cos(theta)
        elif F == 3:
            return -np.sin(theta)
        else:
            print('ERROR IN Theta WHEN CONSTRUCTING BOIUNDARY CONVECTION MATRIX')
            
    def get_S_quad(cell):
        # Returns which angular quadrant a cell is in
        [a0, a1] = cell.pos
        S_quads = [[0, np.pi/2], [np.pi/2, np.pi],
               [np.pi, 3*np.pi/2], [3*np.pi/2, 2*np.pi]]
        for SS in range(0, 4):
            S_quad = S_quads[SS]
            if (a0 >= S_quad[0]) and (a1 <= S_quad[1]):
                # Careful here with comparing floats
                return SS
        
    
    # Store local-column matrices in here
    # We have to assemble a lot on inter-column interaction matrices,
    # so the construction is a bit more difficult.
    col_keys = mesh.cols.keys()
    ncols = len(col_keys)
    col_idxs = dict()
    for C in range(0, ncols):
        col_idxs[col_keys[C]] = C
    col_mtxs = [[None] * ncols for C in range(0, ncols)]
    
    # The local-column matrices come in two kinds: M^CC and M^CC'.
    # The M^CC have to be constructed in four parts: M^CC_F.
    # The M^CC' can be constructed in one part.
    # We loop through each column C, then through each face F of C.
    # For each face, loop through each element K of C.
    # Depending on K, we contribute to M^CC_F or M^CC'.
    # Hold all four M^CC_F, add them together after all of the loops.
    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            # Get column information
            [x0, y0, x1, y1, dx, dy, dof_x, dof_y, weights_x, weights_y] = \
                get_col_info(col)
            col_idx_0 = col_idxs[col_key]

            # Set up cell indexing for column matrices
            cell_keys_0 = sorted(col.cells.keys())
            ncells_0 = len(cell_keys_0)
            cell_idxs_0 = dict()
            for K in range(0, ncells_0):
                cell_idxs_0[cell_keys_0[K]] = K
            cell_mtxs_00 = [None] * ncells_0 # Intra-column matrices are block-diagonal

            # Loop through the faces of C
            for F in range(0, 4): # For each face of C, find the neighboring columns
                axis = F%2
                col_nhbr_loc = nhbr_locs[int(F/2)]
                [flag, col_nhbr_1, col_nhbr_2] = \
                    ji_mesh.get_col_nhbr(mesh, col,
                                         axis = axis,
                                         nhbr_loc = nhbr_loc)
                if flag == 'nn':
                    # In the no-neighbor case, all local-element matrices M^KK
                    # are in the local-column matrix M^CC.
                    for cell_key, cell in sorted(col.cells.items()):
                        if cell.is_lf:
                            S_quad = get_S_quad(cell)
                            [a0, a1, da, dof_a, nodes_a, weights_a] = \
                                get_cell_info(cell)
                            cell_idx_0 = cell_idxs_0[cell_keys_0]

                            # Indexing from p, q, r to alpha
                            # Is the same for i, j, a to beta
                            def alpha(pp, qq, rr):
                                val = dof_a * dof_y * pp \
                                    + dof_a * qq \
                                    + rr
                                return val

                            th = norm_to_local(a0, a1, nodes_a)

                            # Delta_ coefficient depends on F
                            dcoeffs = [dy * da / 4, -dx * da / 4, -dy * da / 4,
                                       dx * da / 4]
                            dcoeff = dcoeffs[F]
                            # cell matrix size depends on F%2
                            cell_F_ndofs = [dof_y * dof_a, dof_x * dof_a]
                            cell_F_ndof = cell_F_ndofs[F%2]

                            alphalist = np.zeros([cell_F_ndof],
                                                 dtype = np.int32) # alpha index
                            betalist = np.zeros([cell_F_ndof],
                                                dtype = np.int32) # beta index
                            vlist = np.zeros([cell_F_ndof]) # Matrix entry
                            cnt = 0
                            
                            if (F == 0):
                                for jj in range(0, dof_y):
                                    wy_j = weights_y[jj]
                                    for aa in range(0, dof_a):
                                        wth_a = weights_a[aa]
                                        Theta_F_a = Theta(F, th[aa])
                                        # Index of entry
                                        alphalist[cnt] = alpha(dof_x - 1, jj, aa)
                                        betalist[cnt] = beta(dof_x - 1, jj, aa)
                                        
                                        vlist[cnt] = dcoeff * wy_j * wth_a * Theta_F_a
                                        cnt += 1
                            elif (F == 1):
                                for ii in range(0, dof_x):
                                    wx_i = weights_x[ii]
                                    for aa in range(0, dof_a):
                                        wth_a = weights_a[aa]
                                        Theta_F_a = Theta(F, th[aa])
                                        # Index of entry
                                        alphalist[cnt] = alpha(ii, dof_y - 1, aa)
                                        betalist[cnt] = beta(ii, dof_y - 1, aa)
                                        
                                        vlist[cnt] = dcoeff * wx_i * wth_a * Theta_F_a
                                        cnt += 1
                            elif (F == 2):
                                for jj in range(0, dof_y):
                                    wy_j = weights_y[jj]
                                    for aa in range(0, dof_a):
                                        wth_a = weights_a[aa]
                                        Theta_F_a = Theta(F, th[aa])
                                        # Index of entry
                                        alphalist[cnt] = alpha(0, jj, aa)
                                        betalist[cnt] = beta(0, jj, aa)
                                        
                                        vlist[cnt] = dcoeff * wy_j * wth_a * Theta_F_a
                                        cnt += 1
                            elif (F == 3):
                                for ii in range(0, dof_x):
                                    wx_i = weights_x[ii]
                                    for aa in range(0, dof_a):
                                        wth_a = weights_a[aa]
                                        Theta_F_a = Theta(F, th[aa])
                                        # Index of entry
                                        alphalist[cnt] = alpha(ii, 0, aa)
                                        betalist[cnt] = beta(ii, 0, aa)
                                        
                                        vlist[cnt] = dcoeff * wx_i * wth_a * Theta_F_a
                                        cnt += 1
                            else:
                                print('RTDG_AMR ERROR: F outside of valid range!')

                            if cell_mtxs_00[cell_idx_0] == None:
                                cell_mtxs_00[cell_idx_0] = coo_matrix((vlist,
                                                                       (alphalist, betalist)))
                            else:
                                cell_mtxs_00[cell_idx_0] += coo_matrix((vlist,
                                                                       (alphalist, betalist)))
                else: # We also have to construct the inter-cloumn matrices M^CC'
                    for nhbr_col in [col_nhbr_1, col_nhbr_2]:
                        # Set up cell indexing for column matrices
                        cell_keys_1 = sorted(nhbr_col.cells.keys())
                        ncells_1 = len(cell_keys_1)
                        cell_idxs_1 = dict()
                        for K in range(0, ncells_1):
                            cell_idxs_1[cell_keys_1[K]] = K
                        cell_mtxs_01 = [[None] * ncells_1 for K in range(0, ncells_0)]
                                            # Inter-column matrices are not block-
                                            # diagonal
                        for cell_key, cell in sorted(col.cells.items()):
                            if cell.is_lf:
                                S_quad = get_S_quad(cell)
                                [a0, a1, da, dof_a, nodes_a, weights_a] = \
                                    get_cell_info(cell)
                                cell_idx_0 = cell_idxs_0[cell_keys_0]
                                
                                # Find if need neighbor information
                                
                                if ((S_quad == 0 and (F == 0 or F == 1))
                                    or (S_quad == 1 and (F == 1 or F == 2))
                                    or (S_quad == 2 and (F == 2 or F == 3))
                                    or (S_quad == 3 and (F == 3 or F == 0))):
                                    # In F+, use information from self,
                                    # I think this should look similar to flag == nn case?
                                    # CONTINUE FROM HERE
                                
                                    # Indexing from p, q, r to alpha
                                    # Is the same for i, j, a to beta
                                    def alpha(pp, qq, rr):
                                        val = dof_a * dof_y * pp \
                                            + dof_a * qq \
                                            + rr
                                        return val
                                    
                                    th = norm_to_local(a0, a1, nodes_a)
                                    
                                    # Delta_ coefficient depends on F
                                    dcoeffs = [dy * da / 4, -dx * da / 4, -dy * da / 4,
                                               dx * da / 4]
                                    dcoeff = dcoeffs[F]
                                    # cell matrix size depends on F%2
                                    cell_F_ndofs = [dof_y * dof_a, dof_x * dof_a]
                                    cell_F_ndof = cell_F_ndofs[F%2]
                                    
                                    alphalist = np.zeros([cell_F_ndof],
                                                         dtype = np.int32) # alpha index
                                    betalist = np.zeros([cell_F_ndof],
                                                        dtype = np.int32) # beta index
                                    vlist = np.zeros([cell_F_ndof]) # Matrix entry
                                    cnt = 0
                                    
                                    if (F == 0):
                                        for jj in range(0, dof_y):
                                            wy_j = weights_y[jj]
                                            for aa in range(0, dof_a):
                                                wth_a = weights_a[aa]
                                                Theta_F_a = Theta(F, th[aa])
                                                # Index of entry
                                                alphalist[cnt] = alpha(dof_x - 1, jj, aa)
                                                betalist[cnt] = beta(dof_x - 1, jj, aa)
                                                
                                                vlist[cnt] = dcoeff * wy_j * wth_a * Theta_F_a
                                                cnt += 1
                                    elif (F == 1):
                                        for ii in range(0, dof_x):
                                            wx_i = weights_x[ii]
                                            for aa in range(0, dof_a):
                                                wth_a = weights_a[aa]
                                                Theta_F_a = Theta(F, th[aa])
                                                # Index of entry
                                                alphalist[cnt] = alpha(ii, dof_y - 1, aa)
                                                betalist[cnt] = beta(ii, dof_y - 1, aa)
                                                
                                                vlist[cnt] = dcoeff * wx_i * wth_a * Theta_F_a
                                                cnt += 1
                                    elif (F == 2):
                                        for jj in range(0, dof_y):
                                            wy_j = weights_y[jj]
                                            for aa in range(0, dof_a):
                                                wth_a = weights_a[aa]
                                                Theta_F_a = Theta(F, th[aa])
                                                # Index of entry
                                                alphalist[cnt] = alpha(0, jj, aa)
                                                betalist[cnt] = beta(0, jj, aa)
                                                
                                                vlist[cnt] = dcoeff * wy_j * wth_a * Theta_F_a
                                                cnt += 1
                                    elif (F == 3):
                                        for ii in range(0, dof_x):
                                            wx_i = weights_x[ii]
                                            for aa in range(0, dof_a):
                                                wth_a = weights_a[aa]
                                                Theta_F_a = Theta(F, th[aa])
                                                # Index of entry
                                                alphalist[cnt] = alpha(ii, 0, aa)
                                                betalist[cnt] = beta(ii, 0, aa)
                                                
                                                vlist[cnt] = dcoeff * wx_i * wth_a * Theta_F_a
                                                cnt += 1
                                    else:
                                        print('RTDG_AMR ERROR: F outside of valid range!')
                                    
                                    if cell_mtxs_00[cell_idx_0] == None:
                                        cell_mtxs_00[cell_idx_0] = coo_matrix((vlist,
                                                                               (alphalist, betalist)))
                                    else:
                                        cell_mtxs_00[cell_idx_0] += coo_matrix((vlist,
                                                                                (alphalist, betalist)))


                                        ## Extra from intr_conv matrix
            # Store local-element matrices in here
            ncells = len(list(col.cells.keys()))
            cell_mtxs = ncells * [ncells * [0]]

            # _0 refers to element K' in the equations
            # _1 refers to element K in the equations
            cell_idx_0 = 0
            for cell_key_0, cell_0 in sorted(col.cells.items()):
                if cell_0.is_lf:
                    [dof_a_0] = cell_0.ndofs

                    # Indexing from p, q, r to alpha
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
