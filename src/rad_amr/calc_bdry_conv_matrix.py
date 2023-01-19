import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag, bmat

from .Projection import Projection_2D
from .matrix_utils import push_forward, pull_back, get_col_info, get_cell_info

import dg.quadrature as qd
from dg.mesh import ji_mesh, tools

def calc_bdry_conv_matrix(mesh):

    # Variables that are the same throughout the loops
    nhbr_locs = ['+', '-']
        
    
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
                    col_mtxs[col_idx_0] = calc_nn_col_mtx(col, F, cell_idxs_0):
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
                                    # In F+, use information from self, i.e.,
                                    # contribute to M^CC
                                    # I think this should look similar to flag == nn case?
                                    # CONTINUE FROM HERE
                                
                                    # Indexing from p, q, r to alpha
                                    # Is the same for i, j, a to beta
                                    def alpha(pp, qq, rr):
                                        val = dof_a * dof_y * pp \
                                            + dof_a * qq \
                                            + rr
                                        return val
                                    
                                    th = push_forward(a0, a1, nodes_a)
                                    
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
                                                betalist[cnt] = alpha(dof_x - 1, jj, aa)
                                                
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
                                                betalist[cnt] = alpha(ii, dof_y - 1, aa)
                                                
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
                                                betalist[cnt] = alpha(0, jj, aa)
                                                
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
                                                betalist[cnt] = alpha(ii, 0, aa)
                                                
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
                    th_0 = push_forward(a0_0, a1_0, nodes_a_0)

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
                            th_1 = push_forward(a0_1, a1_1, nodes_a_1)

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

def calc_nn_col_mtx(col, F, cell_idxs):
    """
    Create the intra-column matrix for the no-neighbor case.
    """
    
    # Get column information
    [~, ~, ~, ~, dx, dy, dof_x, dof_y, weights_x, weights_y] = get_col_info(col)
    
    # Set up cell matrices
    ncells = len(col.cells.keys())
    cell_mtxs = [None] * ncells # Intra-column matrices are block-diagonal

    # Construct the cell matrices
    for cell_key, cell in sorted(col.cells.items()):
        if cell.is_lf:
            # Get cell information
            cell_idx = cell_idxs[cell_key]
            
            [a0, a1, da, dof_a, nodes_a, weights_a] = \
                get_cell_info(cell)
            th = push_forward(a0, a1, nodes_a)
            
            # Indexing from p, q, r to alpha
            # Is the same for i, j, a to beta
            def alpha(pp, qq, rr):
                val = dof_a * dof_y * pp \
                    + dof_a * qq \
                    + rr
                return val

            # Construct the sparse cell matrix
            if (F == 0):
                Theta_F = np.cos(th)
                dcoeff  = dy * da / 4
                ndof    = dof_y * dof_a
                idx     = 0

                # We have alpha = beta so we skip making betalist
                alphalist = np.zeros([ndof], dtype = np.int32) # alpha index
                vlist = np.zeros([ndof]) # Entry value
                
                for jj in range(0, dof_y):
                    wy_j = weights_y[jj]
                    for aa in range(0, dof_a):
                        wth_a = weights_a[aa]
                        Theta_F_a = Theta_F[aa]

                        # Because dirac-deltas, we have
                        # i = p = f, j = q, a = r, so alpha = beta
                        alphalist[idx] = alpha(dof_x - 1, jj, aa)
                        
                        vlist[idx] = dcoeff * wy_j * wth_a * Theta_F_a
                        idx += 1
                        
            elif (F == 1):
                Theta_F = np.sin(th)
                dcoeff  = -dx * da / 4
                ndof    = dof_x * dof_a
                idx     = 0

                # We have alpha = beta so we skip making betalist
                alphalist = np.zeros([ndof], dtype = np.int32) # alpha index
                vlist = np.zeros([ndof]) # Entry value
                
                for ii in range(0, dof_x):
                    wx_i = weights_x[ii]
                    for aa in range(0, dof_a):
                        wth_a = weights_a[aa]
                        Theta_F_a = Theta_F[aa]

                        # Because dirac-deltas, we have
                        # i = p, j = q = f, a = r, so alpha = beta
                        alphalist[idx] = alpha(ii, dof_y - 1, aa)
                        
                        vlist[idx] = dcoeff * wx_i * wth_a * Theta_F_a
                        idx += 1
                        
            elif (F == 2):
                Theta_F = -np.cos(th)
                dcoeff  = -dy * da / 4
                ndof    = dof_y * dof_a
                idx     = 0

                # We have alpha = beta so we skip making betalist
                alphalist = np.zeros([ndof], dtype = np.int32) # alpha index
                vlist = np.zeros([ndof]) # Entry value
                
                for jj in range(0, dof_y):
                    wy_j = weights_y[jj]
                    for aa in range(0, dof_a):
                        wth_a = weights_a[aa]
                        Theta_F_a = Theta_F[aa]

                        # Because dirac-deltas, we have
                        # i = p = 0, j = q, a = r, so alpha = beta
                        alphalist[idx] = alpha(0, jj, aa)
                        
                        vlist[idx] = dcoeff * wy_j * wth_a * Theta_F_a
                        idx += 1
                        
            elif (F == 3):
                Theta_F = -np.sin(th)
                dcoeff  = dx * da / 4
                ndof    = dof_x * dof_a
                idx     = 0

                # We have alpha = beta so we skip making betalist
                alphalist = np.zeros([ndof], dtype = np.int32) # alpha index
                vlist = np.zeros([ndof]) # Entry value
                
                for ii in range(0, dof_x):
                    wx_i = weights_x[ii]
                    for aa in range(0, dof_a):
                        wth_a = weights_a[aa]
                        Theta_F_a = Theta_F[aa]

                        # Because dirac-deltas, we have
                        # i = p, j = q = 0, a = r, so alpha = beta
                        alphalist[idx] = alpha(ii, 0, aa)
                        
                        vlist[idx] = dcoeff * wx_i * wth_a * Theta_F_a
                        cnt += 1
                        
            else:
                print('RTDG_AMR ERROR: F outside of valid range!')
                
            cell_mtxs[cell_idx] = coo_matrix((vlist, (alphalist, alphalist)))

    nn_col_mtx = block_diag(cell_mtxs, format = 'csr')

    return nn_col_mtx


def calc_yn_col_mtxs(col_0, nhbrs, F, cell_idxs_0, nhbr_cell_idxs):
    """
    Create the intra- and inter-column matrices for the yes-neighbor case.
    """

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

    # 2023/01/18 - First pass, making intra-column matrix
    # Get column, neighbor information
    [~, ~, ~, ~, dx_0, dy_0, dof_x_0, dof_y_0,
     weights_x_0, weights_y_0] = get_col_info(col_0)
    ncells_0 = len(col_0.cells.keys())

    # Set up cell matrices
    cell_mtxs_00 = [None] * ncells_0 # Intra-column matrices are block-diagonal

    # Do the same for neighbors and inter-column matrices
    nnhbrs = len(nhbrs)
    nhbr_info = [None] * nnhbrs
    nhbr_ncells = [None] * nnhbrs
    cell_mtxs_0n = [None] * nnhbrs
    for nn in range(0, nnhbrs):
        nhbr = nhbrs[nn]
        if nhbr:
            if nhbr.is_lf:
                nhbr_info[nn] = get_col_info(nhbrs[nn])
                nhbr_ncells[nn] = len(nhbr.cells.keys())
                cell_mtxs_0n[nn] = [[None] * nhbr_ncells[nn]
                                    for K in range(0, ncells_0)]

    # Construct the cell matrices
    for cell_key_0, cell_0 in sorted(col.cells.items()):
        if cell_0.is_lf:
            # Get cell information
            S_quad_0 = get_S_quad(cell_0)
            cell_idx_0 = cell_idxs[cell_key_0]

            # thb is short for \overline{theta}, the pull-back nodal basis
            # for cell_0
            [a0_0, a1_0, da_0, dof_a_0, thb_0, weights_a_0] = \
                get_cell_info(cell_0)
            
            # Indexing from p, q, r to alpha
            def alpha(pp, qq, rr):
                val = dof_a_0 * dof_y_0 * pp \
                    + dof_a_0 * qq \
                    + rr
                return val

            # Det. if in F+ (Fp) or F- (Fm)
            is_Fp = ((S_quad_0 == 0 and (F == 0 or F == 1))
                     or (S_quad_0 == 1 and (F == 1 or F == 2))
                     or (S_quad_0 == 2 and (F == 2 or F == 3))
                     or (S_quad_0 == 3 and (F == 3 or F == 0)))
            if is_Fp:
                # Construct the sparse cell matrix
                # th is the push-forward of thb
                th_0 = push_forward(a0_0, a1_0, thb_0)
                if (F == 0):
                    Theta_F = np.cos(th_0)
                    dcoeff  = dy_0 * da_0 / 4
                    ndof_0  = dof_y_0 * dof_a_0
                    idx     = 0
                    
                    # We have alpha = beta so we skip making betalist
                    alphalist = np.zeros([ndof_0], dtype = np.int32) # alpha index
                    vlist = np.zeros([ndof_0]) # Entry value
                    
                    for jj in range(0, dof_y_0):
                        wy_j = weights_y_0[jj]
                        for aa in range(0, dof_a_0):
                            wth_a = weights_a_0[aa]
                            Theta_F_a = Theta_F[aa]
                            
                            # Because dirac-deltas, we have
                            # i = p = f, j = q, a = r, so alpha = beta
                            alphalist[idx] = alpha(dof_x_0 - 1, jj, aa)
                            
                            vlist[idx] = dcoeff * wy_j * wth_a * Theta_F_a
                            idx += 1
                            
                elif (F == 1):
                    Theta_F = np.sin(th_0)
                    dcoeff  = -dx_0 * da_0 / 4
                    ndof_0  = dof_x_0 * dof_a_0
                    idx     = 0
                    
                    # We have alpha = beta so we skip making betalist
                    alphalist = np.zeros([ndof_0], dtype = np.int32) # alpha index
                    vlist = np.zeros([ndof_0]) # Entry value
                    
                    for ii in range(0, dof_x_0):
                        wx_i = weights_x_0[ii]
                        for aa in range(0, dof_a_0):
                            wth_a = weights_a_0[aa]
                            Theta_F_a = Theta_F[aa]
                            
                            # Because dirac-deltas, we have
                            # i = p, j = q = f, a = r, so alpha = beta
                            alphalist[idx] = alpha(ii, dof_y_0 - 1, aa)
                            
                            vlist[idx] = dcoeff * wx_i * wth_a * Theta_F_a
                            idx += 1
                            
                elif (F == 2):
                    Theta_F = -np.cos(th_0)
                    dcoeff  = -dy_0 * da_0 / 4
                    ndof_0  = dof_y_0 * dof_a_0
                    idx     = 0
                    
                    # We have alpha = beta so we skip making betalist
                    alphalist = np.zeros([ndof_0], dtype = np.int32) # alpha index
                    vlist = np.zeros([ndof_0]) # Entry value
                    
                    for jj in range(0, dof_y_0):
                        wy_j = weights_y_0[jj]
                        for aa in range(0, dof_a_0):
                            wth_a = weights_a_0[aa]
                            Theta_F_a = Theta_F[aa]
                            
                            # Because dirac-deltas, we have
                            # i = p = 0, j = q, a = r, so alpha = beta
                            alphalist[idx] = alpha(0, jj, aa)
                            
                            vlist[idx] = dcoeff * wy_j * wth_a * Theta_F_a
                            idx += 1
                            
                elif (F == 3):
                    Theta_F = -np.sin(th_0)
                    dcoeff  = dx_0 * da_0 / 4
                    ndof_0  = dof_x_0 * dof_a_0
                    idx     = 0
                    
                    # We have alpha = beta so we skip making betalist
                    alphalist = np.zeros([ndof_0], dtype = np.int32) # alpha index
                    vlist = np.zeros([ndof_0]) # Entry value
                    
                    for ii in range(0, dof_x_0):
                        wx_i = weights_x_0[ii]
                        for aa in range(0, dof_a_0):
                            wth_a = weights_a_0[aa]
                            Theta_F_a = Theta_F[aa]
                            
                            # Because dirac-deltas, we have
                            # i = p, j = q = 0, a = r, so alpha = beta
                            alphalist[idx] = alpha(ii, 0, aa)
                            
                            vlist[idx] = dcoeff * wx_i * wth_a * Theta_F_a
                            cnt += 1
                            
                else:
                    print('RTDG_AMR ERROR: F outside of valid range!')
                    
                cell_mtxs_00[cell_idx] = coo_matrix((vlist, (alphalist, alphalist)))

            else: # Not in Fp, use neighbor values
                for nn in range(0, nnhbrs):
                    nhbr = nhbrs[nn]
                    if nhbr:
                        if nhbr.is_lf:
                            [~, ~, ~, ~, dx_1, dy_1, dof_x_1, dof_y_1,
                             weights_x_1, weights_y_1] = nhbr_info[nn]
                            nhbr_cells = get_cell_nhbr_in_col(cell_0, nhbr)
                            for nhbr_cell in nhbr_cells:
                                cell_key_1 = nhbr_cell.key
                                cell_idx_1 = nhbr_cell_idxs[nn][cell_key_1]
                                
                                [a0_1, a1_1, da_1,
                                 dof_a_1, thb_1, weights_a_1] = \
                                     get_cell_info(nhbr_cell)
                                
                                # Indexing from i, j, a to beta
                                def beta(ii, jj, aa):
                                    val = dof_a_1 * dof_y_1 * ii \
                                        + dof_a_1 * jj \
                                        + aa
                                    return val

                                th_0 = push_forward(a0_0, a1_0, thb_0)
                                th_1 = push_forward(a0_1, a1_1, thb_1)
                                
                                if (F == 0):
                                    def E_theta(dof_a_0, dof_a_1):
                                        if dof_a_0 >= dof_a_1:
                                            thb_0_1 = pull_back(a0_1, a1_1, th_0))
                                            Theta_F = np.cos(th_0)

                                            out = np.zeros([dof_a_1, dof_a_0])
                                            for rr in range(0, dof_a_0):
                                                wth_r = weights_th_0[rr]
                                                Theta_F_r = Theta_F[rr]
                                                for aa in range(0, dof_a_1):
                                                    xi_a = gl_eval(thb_1, aa,
                                                                   thb_0_1[rr])
                                                    out[aa, rr] = w_th_r * Theta_F_r * xi_a
                                        elif dof_a_0 < dof_a_1:
                                            th_1_0 = push_forward(a0_0, a1_0, thb_1)
                                            thb_1_0_1 = pull_back(a0_1, a1_1, th_1_0)
                                            Theta_F = np.cos(th_1_0)

                                            out = np.zeros([dof_a_1, dof_a_0])
                                            for rr in range(0, dof_a_0):
                                                xi_r = gl_eval(thb_0, rr, thb_1)
                                                for aa in range(0, dof_a_1):
                                                    xi_a = gl_eval(thb_1, aa,
                                                                   thb_1_0_1)
                                                    out[aa, rr] = np.sum(weights_th_0 * Theta_F * xi_a * xi_r)
                
    col_mtx_00 = block_diag(cell_mtxs_00, format = 'csr')

    return nn_col_mtx
