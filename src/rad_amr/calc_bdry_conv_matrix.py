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
                    col_mtx_00 = calc_nn_col_mtx(col, F, cell_idxs_0):
                    # The intera-column matrix may already exist. If so, add to it.
                    if col_mtxs[col_idx_0][col_idx_0]:
                        col_mtxs[col_idx_0][col_idx_0] += col_mtx_00
                    else:
                        col_mtxs[col_idx_0][col_idx_0] = col_mtx_00
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

                        [col_mtx_00, col_mtxs[col_idx_0][col_idx_1]] = \
                            calc_yn_col_mtxs(col_0, nhbr_col, F, cell_idxs_0, cell_idxs_1)
                        if col_mtxs[col_idx_0][col_idx_0]:
                            col_mtxs[col_idx_0][col_idx_0] += col_mtx_00
                        else:
                            col_mtxs[col_idx_0][col_idx_0] = col_mtx_00
                
    M_bdry_conv = bmat(col_mtxs, format = 'csr')

    return M_bdry_conv

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


def calc_yn_col_mtxs(col_0, col_1, F, cell_idxs_0, cell_idxs_1):
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

        return None

    # Get information about column C
    # _0 => Cell K in equations (in column C)
    # **b => Pull back coordinates (in [-1, 1])
    # w* => Quadrature weights
    [x0_0, y0_0, x1_0, y1_0]         = col_0.pos
    [dx_0, dy_0]                     = [x1_0 - x0_0, y1_0, y0_0]
    [ndof_x_0, ndof_y_0]             = col_0.ndofs
    [xxb_0, wx_0, yyb_0, wy_0, ~, ~] = qd.quad_xya(ndof_x_0, ndof_y_0, 1)
    ncells_0 = len(col_0.cells.keys())

    # Set array to store cell matrices for intra-column matrix
    # Are block-diagonal since cells within a column don't interact
    cell_mtxs_00 = [None] * ncells_0
    
    # Get information about neighboring column C'
    # Store it to access from loop later
    # _1 => Cell K^(n) in equations (in neighboring column C')
    nnhbrs = 0
    for nn in range(0, len(nhbrs)):
        if nhbrs[nn]:
            nnhbrs += 1
        
    nhbr_info = [None] * nnhbrs

    # Set array to stroe cell matrices for inter-column matrices
    # Certainly not block diagonal, but set sizes later once we get number of
    # cells in neighboring columns
    [x0_1, y0_1, x1_1, y1_1]         = col_1.pos
    [dx_1, dy_1]                     = [x1_1 - x0_1, y1_1, y0_1]
    [ndof_x_1, ndof_y_1]             = col_1.ndofs
    [xxb_1, wx_1, yyb_1, wy_1, ~, ~] = qd.quad_xya(ndof_x_1, ndof_y_1, 1)
    ncells_1 = len(col_1.cells.keys())
    
    cell_mtxs_01 = [[None] * ncells_1 for K in range(0, ncells_0)]
    
                
    # Loop through cells of column C
    # For each cell in column C, we loop through the neighboring cells K^(n)
    # in neighboring column C'
    for cell_key_0, cell_0 in sorted(col_0.cells.items()):
        if cell_0.is_lf:
            # Get information about cell K in column C
            S_quad_0 = get_S_quad(cell_0)
            cell_idx_0 = cell_idxs[cell_key_0] # Matrix index of cell 0 in
                                               # column matrices
            [th0_0, th1_0]             = cell_0.pos
            dth_0                      = th1_0 - th0_0
            ndof_th_0                  = cell_0.ndofs[0]
            [~, ~, ~, ~, thb_0, wth_0] = qd.quad_xya(1, 1, ndof_th_0)
            
            # (p, q, r) => alpha
            # alpha is number of rows, always corresponds to K
            def alpha(pp, qq, rr):
                out = ndof_th_0 * ndof_y_0 * pp \
                    + dof_th_0 * qq \
                    + rr
                return out

            # Det. if in F+ (Fp) or F- (Fm)
            is_Fp = ((S_quad_0 == 0 and (F == 0 or F == 1))
                     or (S_quad_0 == 1 and (F == 1 or F == 2))
                     or (S_quad_0 == 2 and (F == 2 or F == 3))
                     or (S_quad_0 == 3 and (F == 3 or F == 0)))
            
            if is_Fp:
                # In Fp, we only construct the intra-column cell matrix

                # Many quantities are actually dependent only on the parity of F
                # so we can separate primarily into two cases.
                # **f => Push forward coordinates (in [*0, *1])
                thf_0 = push_forward(th0_0, th1_0, thb_0)

                # Some parameters are dependent on the parity of F
                # Which we handle here
                if (F%2 == 0):
                    # Although Theta_F and dcoeff are dependent on F,
                    # their product is only dependent on the parity of F
                    # The equations here don't match the documentation, but
                    # their product will
                    Theta_F_0 = np.cos(thf_0)
                    dcoeff_0  = dy_0 * dth_0 / 4

                    ndof_0 = ndof_y_0 * ndof_th_0

                    # The entry locations are mostly dependent on only the
                    # parity of F, and their value is entirely dependent on F,
                    # so we can handle those here, too
                    # We have alpha = beta so we skip making betalist
                    alphalist = np.zeros([ndof_0], dtype = np.int32) # alpha index
                    vlist = np.zeros([ndof_0]) # Entry value

                    if (F == 0):
                        x_idx = ndof_x_0 - 1
                    elif (F == 2):
                        x_idx = 0
                    else:
                        print('ERROR: BDRY CONV MTX F%2 == 0')
                        quit()
                    
                    idx = 0
                    for jj in range(0, ndof_y_0):
                        wy_0_j = wy_0[jj]
                        for aa in range(0, ndof_th_0):
                            wth_0_a = wth_0[aa]
                            Theta_F_0_a = Theta_F_0[aa]

                            alphalist[idx] = alpha(x_idx, jj, aa)
                            vlist[idx]  = dcoeff_0 * wy_0_j * wth_0_a * Theta_F_0_a

                            idx += 1
                    
                elif (F%2 == 1):
                    Theta_F_0 = np.sin(thf_0)
                    dcoeff_0  = dx_0 * dth_0 / 4

                    ndof_0    = -ndof_x_0 * ndof_th_0

                    alphalist = np.zeros([ndof_0], dtype = np.int32) # alpha index
                    vlist = np.zeros([ndof_0]) # Entry value

                    if (F == 1):
                        y_idx = ndof_y_0 - 1
                    elif (F == 3):
                        y_idx = 0
                    else:
                        print('ERROR: BDRY CONV MTX F%2 == 1')
                        quit()
                        
                    idx = 0
                    for ii in range(0, ndof_x_0):
                        wx_0_i = wx_0[ii]
                        for aa in range(0, ndof_th_0):
                            wth_0_a = wth_0[aa]
                            Theta_F_0_a = Theta_0_F[aa]

                            alphalist[idx] = alpha(ii, y_idx, aa)
                            vlist[idx]  = dcoeff_0 * wx_0_i * wth_0_a * Theta_F_0_a
                            
                            cnt += 1
                    
                cell_mtxs_00[cell_idx_0] = coo_matrix((vlist, (alphalist, alphalist)))

            else:
                # In Fm, we construct the inter-column cell matrices
                nhbr_cells = get_cell_nhbr_in_col(cell_0, col_1)
                for cell_1 in nhbr_cells:
                    cell_key_1 = cell_1.key
                    cell_idx_1 = nhbr_cell_idxs[cell_key_1]
                    
                    [th0_1, th1_1] = cell_1.pos
                    dth_1          = th1_1 - th0_1
                    ndof_th_1      = cell_1.ndofs[0]
                    [~, ~, ~, ~, thb_1, wth_1] = \
                        qd.quad_xya(1, 1, ndof_th_1)
                    
                    # (i, j, a) => beta
                    # beta is number of columns
                    # in Fm, beta always corresponds to K^(n)
                    def beta(ii, jj, aa):
                        out = ndof_th_1 * ndof_y_1 * ii \
                            + ndof_th_1 * jj \
                            + aa
                        return out
                    
                    # REMINDER: **f => Push forward coordinates
                    # (in [*0, *1])
                    thf_0 = push_forward(th0_0, th1_0, thb_0)
                    thf_1 = push_forward(th0_1, th1_1, thb_1)
                    
                    # Many quantities are actually dependent only on
                    # the parity of F not F itself, so we can 
                    # separate primarily into two cases
                    
                    if (F%2 == 0):
                        # Although E_theta and dcoeff are dependent
                        # on F, their product is only dependent on
                        # the parity of F.
                        # The equations here don't match the
                        # documentation, but their product will
                        
                        # Construct E^(K^(n)K,theta)
                        E_th = np.zeros([ndof_th_0, ndof_th_1])
                        if ndof_th_0 >= ndof_th_1:
                            Theta_F_0 = np.cos(thf_0)
                            thb_0_1 = pull_back(th0_1, th1_1, thf_0))
                            
                            for aa in range(0, ndof_th_1):
                                for rr in range(0, ndof_th_0):
                                    wth_0_r = wth_0[rr]
                                    Theta_F_0_r = Theta_F[rr]
                                    xi_a = gl_eval(thb_1, aa, thb_0_1[rr])
                                    E_th[aa, rr] = wth_0_r * Theta_F_0_r * xi_a
                                    
                        elif ndof_th_0 < ndof_th_1:
                            thf_1_0 = push_forward(th0_0, th1_0, thb_1)
                            Theta_F = np.cos(thf_1_0)
                            
                            thb_1_0_1 = pull_back(th0_1, th1_1, thf_1_0)
                            
                            for aa in range(0, ndof_th_1):
                                for rr in range(0, ndof_th_0):
                                    for aap in range(0, ndof_th_1):
                                        xi_a = gl_eval(thb_1, aa, thb_1_0_1[aap])
                                        xi_r = gl_eval(thb_0, rr, thb_1[aap])
                                        E_th[aa, rr] += wth_1[aap] * Theta_F[aap] * xi_a * xi_r
                                        
                        # Construct E^(K^(n)K,y)
                        E_y = np.zeros([ndof_y_1, ndof_y_0])
                        if ndof_y_0 >= ndof_y_1:
                            yyf_0   = push_forward(y0_0, y1_0, yyb_0)
                            yyb_0_1 = pull_back(y0_1, y1_1, yyf_0)
                            
                            for jj in range(0, ndof_y_1):
                                for qq in range(0, ndof_y_0):
                                    wy_0_q = wy_0[qq]
                                    psi_j = gl_eval(yyb_0_1, jj, yyb_0[qq])
                                    E_y[jj, qq] = wy_q * psi_j
                                    
                        elif ndof_y_0 < ndof_y_1:
                            yyf_1_0   = push_forward(y0_0, y1_0, yyb_1)
                            yyb_1_0_1 = pull_back(y0_1, y1_1, yyf_1_0)
                            
                            for jj in range(0, ndof_y_1):
                                for qq in range(0, ndof_y_0):
                                    for jjp in range(0, ndof_y_1):
                                        psi_j = gl_eval(yb_1, jj, yyb_1_0_1[jjp])
                                        psi_q = gl_eval(yyb_0, qq, yyb_1[jjp])
                                        E_y[jj, qq] += wy_1[jjp] * psi_j * psi_q
                                    
                        dcoeff_0  = dy_0 * da_0 / 4
                        
                        ndof_0    = dof_y_0 * dof_th_0
                        ndof_1    = dof_y_1 * dof_th_1
                        
                        # The entry locations are mostly dependent
                        # on only the parity of F, and their value
                        # is entirely dependent on F, so we can
                        # handle those here, too
                        alphalist = np.zeros([ndof_0 * ndof_1], dtype = np.int32) # alpha index
                        betalist  = np.zeros([ndof_0 * ndof_1], dtype = np.int32) # beta index
                        vlist     = np.zeros([ndof_0 * ndof_1]) # Entry value
                        
                        if (F == 0):
                            x_idx_0 = ndof_x_0 - 1
                            x_idx_1 = 1
                        elif (F == 2):
                            x_idx_0 = 0
                            x_idx_1 = ndof_x_1 - 1
                            
                        for jj in range(0, ndof_y_1):
                            for qq in range(0, ndof_y_0):
                                E_y_jq = E_y[jj, qq]
                                for aa in range(0, ndof_th_1):
                                    for rr in range(0, ndof_th_0):
                                        E_th_ar = E_th[aa, rr]
                                        
                                        alphalist[idx] = alpha(x_idx_0, qq, rr)
                                        betalist[idx]  = beta( x_idx_1, jj, aa)
                                        vlist[idx] = dcoeff_0 * E_th_ar * E_y_jq
                                        
                                        idx += 1
                    elif (F%2 == 1):
                        # Construct E^(K^(n)K,theta)
                        E_th = np.zeros([ndof_th_1, ndof_th_0])
                        if ndof_th_0 >= ndof_th_1:
                            Theta_F = np.cos(thf_0)
                            thb_0_1 = pull_back(th0_1, th1_1, thf_0))
                            
                            for aa in range(0, ndof_th_1):
                                for rr in range(0, ndof_th_0):
                                    wth_0_r = wth_0[rr]
                                    Theta_F_r = Theta_F[rr]
                                    xi_a = gl_eval(thb_1, aa,
                                                   thb_0_1[rr])
                                    E_th[aa, rr] = wth_0_r * Theta_F_r * xi_a
                                    
                        elif ndof_th_0 < ndof_th_1:
                            thf_1_0 = push_forward(th0_0, th1_0, thb_1)
                            Theta_F = np.cos(thf_1_0)
                            
                            thb_1_0_1 = pull_back(th0_1, th1_1, thf_1_0)
                            
                            for aa in range(0, ndof_th_1):
                                for rr in range(0, ndof_th_0):
                                    for aap in range(0, ndof_th_1):
                                        xi_a = gl_eval(thb_1, aa, thb_1_0_1[aap])
                                        xi_r = gl_eval(thb_0, rr, thb_1[aap])
                                        E_th[aa, rr] += wth_1[aap] * Theta_F[aap] * xi_a * xi_r
                                        
                        # Construct E^(K^(n)K,x)
                        E_x = np.zeros([ndof_x_1, ndof_x_0])
                        if ndof_x_0 >= ndof_x_1:
                            xxf_0   = push_forward(x0_0, x1_0, xxb_0)
                            xxb_0_1 = pull_back(x0_1, x1_1, xxf_0)
                            
                            for ii in range(0, ndof_x_1):
                                for pp in range(0, ndof_x_0):
                                    wx_0_p = wx_0[pp]
                                    phi_i = gl_eval(xxb_0_1, ii, xxb_0[pp])
                                    E_x[ii, pp] = wx_p * phi_i
                                    
                        elif ndof_x_0 < ndof_x_1:
                            xxf_1_0   = push_forward(x0_0, x1_0, xb_1)
                            xxb_1_0_1 = pull_back(x0_1, x1_1, xf_1_0)
                            
                            for ii in range(0, ndof_x_1):
                                for pp in range(0, ndof_x_0):
                                    for iip in range(0, ndof_x_1):
                                        phi_i = gl_eval(xb_1, jj, xxb_1_0_1[iip])
                                        phi_j = gl_eval(xxb_0, qq, xxb_1[iip])
                                        E_x[jj, qq] += wx_1[iip] * phi_i * phi_p
                                        
                        dcoeff_0  = -dx_0 * da_0 / 4
                        
                        ndof_0    = dof_x_0 * dof_th_0
                        ndof_1    = dof_x_1 * dof_th_1
                        
                        alphalist = np.zeros([ndof_0 * ndof_1], dtype = np.int32) # alpha index
                        betalist  = np.zeros([ndof_0 * ndof_1], dtype = np.int32) # beta index
                        vlist     = np.zeros([ndof_0 * ndof_1]) # Entry value
                        
                        if (F == 1):
                            y_idx_0 = ndof_y_0 - 1
                            y_idx_1 = 0
                        elif (F == 3):
                            y_idx_0 = 0
                            y_idx_1 = ndof_y_1 - 1
                            
                        for ii in range(0, ndof_x_1):
                            for pp in range(0, ndof_x_0):
                                E_x_ip = E_x[ii, pp]
                                for aa in range(0, ndof_th_1):
                                    for rr in range(0, ndof_th_0):
                                        E_th_ar = E_th[aa, rr]
                                        
                                        alphalist[idx] = alpha(pp, y_idx_0, rr)
                                        betalist[idx]  = beta( ii, y_idx_1, aa)
                                        vlist[idx] = dcoeff_0 * E_th_ar * E_x_ip
                                        
                                        idx += 1
                        
                        cell_mtxs_01[cell_idx_0][cell_idx_1] =\
                            coo_matrix((vlist, (alphalist, betalist)))
                
    col_mtx_00 = block_diag(cell_mtxs_00, format = 'csr')
    col_mtx_01 = bmat(cell_mtxs_01, format = 'csr')

    return [col_mtx_00, col_mtx_01]
