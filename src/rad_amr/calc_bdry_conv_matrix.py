import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag, bmat

from .Projection import Projection_2D
from .matrix_utils import push_forward, pull_back

from dg.mesh import ji_mesh
import dg.quadrature as qd

from utils import print_msg


def calc_bdry_conv_matrix(mesh):

    # Variables that are the same throughout the loops
    nhbr_locs = ['+', '-']
    
    # Create column indexing for constructing global mass matrix
    col_idx  = 0
    col_idxs = dict()
    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            col_idxs[col_key] = col_idx
            col_idx += 1
            
    ncols = col_idx # col_idx counts the number of existing columns in mesh
    col_mtxs = [[None] * ncols for C in range(0, ncols)] # We have to assemble a
               # lot on inter-column interaction matrices,
               # so the construction is a bit more difficult.
    
    # The local-column matrices come in two kinds: M^CC and M^CC'.
    # The M^CC have to be constructed in four parts: M^CC_F.
    # The M^CC' can be constructed in one part.
    # We loop through each column C, then through each face F of C.
    # For each face, loop through each element K of C.
    # Depending on K, we contribute to M^CC_F or M^CC'.
    # Hold all four M^CC_F, add them together after all of the loops.
    for col_key_0, col_0 in sorted(mesh.cols.items()):
        if col.is_lf:
            # Use _0 to refer to column C
            # Later, use _1 to refer to column C'
            col_idx_0 = col_idxs[col_key_0]
            
            # Create cell indexing for constructing intra-column
            # boundary convection matrix
            cell_idx_0 = 0
            cell_idxs_0 = dict()
            for cell_key_0, cell_0 in sorted(col_0.cells.items()):
                if cell_0.is_lf:
                    cell_idxs_0[cell_key_0] = cell_idx_0
                    cell_idx_0 += 1

            ncells_0 = cell_idx_0 # cell_idx counts the number of existing cells in column
            cell_mtxs_00 = [None] * ncells_0 # Intra-column boundary convection
                                        # matrix is block-diagonal, and so there
                                        # are only ncell non-zero cell
                                        # boundary convection matrices

            # Loop through the faces of C
            for F in range(0, 4):
                # Find the neighbors of C
                axis = F%2
                nhbr_loc = nhbr_locs[int(F/2)]
                [flag, col_nhbr_1, col_nhbr_2] = \
                    ji_mesh.get_col_nhbr(mesh, col,
                                         axis = axis,
                                         nhbr_loc = nhbr_loc)
                
                # No neighbor, construct intra-column matrix
                if flag == 'nn':
                    col_mtx_00 = calc_nn_col_mtx(col_0, F, cell_idxs_0)
                    
                    # The intra-column matrix may already exist. If so, add to it.
                    if col_mtxs[col_idx_0][col_idx_0] == None:
                        col_mtxs[col_idx_0][col_idx_0] = col_mtx_00
                    else:                        
                        col_mtxs[col_idx_0][col_idx_0] += col_mtx_00

                
                else: # We also have to construct the inter-cloumn matrices M^CC'
                    for col_1 in [col_nhbr_1, col_nhbr_2]:
                        if col_1:
                            if col_1.is_lf:

                                # Due to PBCs, a column can be its own neighbor.
                                # If so, go to no-neighbor case
                                col_key_1 = col_1.key
                                if col_key_0 == col_key_1:
                                    col_mtx_00 = calc_nn_col_mtx(col_0, F, cell_idxs_0)
                                    
                                    # The intra-column matrix may already exist. If so, add to it.
                                    if col_mtxs[col_idx_0][col_idx_0] == None:
                                        col_mtxs[col_idx_0][col_idx_0] = col_mtx_00
                                    else:                        
                                        col_mtxs[col_idx_0][col_idx_0] += col_mtx_00
                                else:
                                    col_idx_1 = col_idxs[col_key_1]
                                    
                                    # Create cell indexing for constructing inter-column
                                    # boundary convection matrix
                                    cell_idx_1 = 0
                                    cell_idxs_1 = dict()
                                    for cell_key_1, cell_1 in sorted(col_1.cells.items()):
                                        if cell_1.is_lf:
                                            cell_idxs_1[cell_key_1] = cell_idx_1
                                            cell_idx_1 += 1
                                            
                                    # Inter-column boundary convection
                                    # matrix is notblock-diagonal, so we
                                    # need room for all possible off-diagonal
                                    # blocks
                                    
                                    
                                    [col_mtx_00, col_mtxs[col_idx_0][col_idx_1]] = \
                                        calc_yn_col_mtxs(col_0, col_1, F,
                                                         cell_idxs_0, cell_idxs_1)

                                    str_0 = str([col_idx_0, col_mtx_00.shape])
                                    str_1 = str([col_idx_0, col_idx_1,
                                                 col_mtxs[col_idx_0][col_idx_1].shape])
                                    
                                    if col_mtxs[col_idx_0][col_idx_0] == None:
                                        col_mtxs[col_idx_0][col_idx_0] = col_mtx_00
                                    else:
                                        col_mtxs[col_idx_0][col_idx_0] += col_mtx_00
                
    # Global boundary convection matrix is not block-diagonal
    # but we arranged the column matrices in the proper form
    bdry_conv_mtx = bmat(col_mtxs, format = 'csr')

    return bdry_conv_mtx

def calc_nn_col_mtx(col, F, cell_idxs):
    """
    Create the intra-column matrix for the no-neighbor case.
    """

    # Since we are only dealing with inter-column stuff, ignore the _0
    # Get column information, quadrature weights
    [x0, y0, x1, y1] = col.pos
    dx = x1 - x0
    dy = y1 - y0
    [ndof_x, ndof_y] = col.ndofs
    
    [_, w_x, _, w_y, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                          nnodes_y = ndof_y)
    
       
    ncells = len(cell_idxs) # cell_idx counts the number of existing cells in column
    cell_mtxs = [None] * ncells # Intra-column matrices are
                                # block-diagonal, and so there are only 
                                # ncell non-zero cell boundary convection
                                # matrices

    # Construct the cell matrices
    for cell_key, cell in sorted(col.cells.items()):
        if cell.is_lf:
            # Get cell information, quadrature weights
            cell_idx     = cell_idxs[cell_key]
            [th0, th1]   = cell.pos
            dth          = th1 - th0
            [ndof_th]    = cell.ndofs
            
            [_, _, _, _, thb, w_th] = qd.quad_xyth(nnodes_th = ndof_th)
            thf = push_forward(th0, th1, thb)
            
            # Indexing from i, j, a to beta
            # Is the same for p, q, r to alpha
            def beta(ii, jj, aa):
                val = ndof_th * ndof_y * ii \
                    + ndof_th * jj \
                    + aa
                return val

            # Some parameters are only dependent on the parity of F
            # Which we handle here
            if (F%2 == 0):
                # Although Theta_F and dcoeff are dependent on F,
                # their product is only dependent on the parity of F
                # The equations here don't match the documentation, but
                # their product will
                Theta_F = np.cos(thf)
                dcoeff  = dy * dth / 4

                ndof = ndof_y * ndof_th # Number of *NON-ZERO* DoFs

                # The entry locations are mostly dependent on only the
                # parity of F, and their value is entirely dependent on F,
                # so we can handle those here, too
                # We have alpha = beta so we skip making alphalist
                betalist = np.zeros([ndof], dtype = np.int32) # beta index
                vlist = np.zeros([ndof]) # Entry value
                
                if (F == 0):
                    x_idx = ndof_x - 1
                elif (F == 2):
                    x_idx = 0
                else:
                    print('ERROR: BDRY CONV MTX F%2 == 0')
                    quit()
                    
                # Construct cell matrix
                idx = 0
                for jj in range(0, ndof_y):
                    wy_j = w_y[jj]
                    for aa in range(0, ndof_th):
                        wth_a = w_th[aa]
                        Theta_F_a = Theta_F[aa]

                        # Because dirac-deltas, we have
                        # i = p = f, j = q, a = r, so alpha = beta
                        betalist[idx] = beta(x_idx, jj, aa)
                        
                        vlist[idx] = dcoeff * wy_j * wth_a * Theta_F_a
                        
                        idx += 1
            
            elif (F%2 == 1):
                # Although Theta_F and dcoeff are dependent on F,
                # their product is only dependent on the parity of F
                # The equations here don't match the documentation, but
                # their product will
                Theta_F = np.sin(thf)
                dcoeff  = -dx * dth / 4

                ndof = ndof_x * ndof_th # Number of *NON-ZERO* DoFs

                # The entry locations are mostly dependent on only the
                # parity of F, and their value is entirely dependent on F,
                # so we can handle those here, too
                # We have alpha = beta so we skip making alphalist
                betalist = np.zeros([ndof], dtype = np.int32) # beta index
                vlist = np.zeros([ndof]) # Entry value
                
                if (F == 1):
                    y_idx = ndof_y - 1
                elif (F == 3):
                    y_idx = 0
                else:
                    print('ERROR: BDRY CONV MTX F%2 == 0')
                    quit()
                    
                # Construct cell matrix
                idx = 0
                for ii in range(0, ndof_x):
                    wx_i = w_x[ii]
                    for aa in range(0, ndof_th):
                        wth_a = w_th[aa]
                        Theta_F_a = Theta_F[aa]

                        # Because dirac-deltas, we have
                        # i = p, j = q = f, a = r, so alpha = beta
                        betalist[idx] = beta(ii, y_idx, aa)
                        
                        vlist[idx] = dcoeff * wx_i * wth_a * Theta_F_a
                        idx += 1
            
            else:
                print('RTDG_AMR ERROR: F outside of valid range!')

            cell_ndof = ndof_x * ndof_y * ndof_th
            cell_mtxs[cell_idx] = coo_matrix((vlist, (betalist, betalist)),
                                             shape = (cell_ndof, cell_ndof))

    nn_col_mtx = block_diag(cell_mtxs, format = 'csr')

    return nn_col_mtx


def calc_yn_col_mtxs(col_0, col_1, F, cell_idxs_0, cell_idxs_1):
    """
    Create the intra- and inter-column matrices for the yes-neighbor case.
    """
    col_ndof_0 = 0
    col_ndof_1 = 0

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
    [dx_0, dy_0]                     = [x1_0 - x0_0, y1_0 - y0_0]
    [ndof_x_0, ndof_y_0]             = col_0.ndofs
    [xxb_0, wx_0, yyb_0, wy_0, _, _] = qd.quad_xyth(nnodes_x = ndof_x_0,
                                                    nnodes_y = ndof_y_0)

    # Create cell indexing for constructing column mass matrix
    ncells_0 = len(cell_idxs_0) # cell_idx counts the number of existing cells in column
    cell_mtxs_00 = [None] * ncells_0 # Intra-column matrices are
                                     # block-diagonal, and so there are only 
                                     # ncell non-zero cell boundary convection
                                     # matrices

    # Set array to store cell matrices for inter-column matrices
    # Certainly not block diagonal, but set sizes later once we get number of
    # cells in neighboring columns
    [x0_1, y0_1, x1_1, y1_1]         = col_1.pos
    [dx_1, dy_1]                     = [x1_1 - x0_1, y1_1 - y0_1]
    [ndof_x_1, ndof_y_1]             = col_1.ndofs
    [xxb_1, wx_1, yyb_1, wy_1, _, _] = qd.quad_xyth(nnodes_x = ndof_x_1,
                                                    nnodes_y = ndof_y_1)
    ncells_1 = len(cell_idxs_1)
    cell_mtxs_01 = [[None] * ncells_1 for K in range(0, ncells_0)]

    # To ensure proper matrix construction, we initialize all cell #
    # matrices to be empty sparse matrices
    for cell_key_0, cell_0 in sorted(col_0.cells.items()):
        if cell_0.is_lf:
            cell_idx_0  = cell_idxs_0[cell_key_0]
            ndof_th_0   = cell_0.ndofs[0]
            cell_ndof_0 = ndof_x_0 * ndof_y_0 * ndof_th_0
            
            for cell_key_1, cell_1 in sorted(col_1.cells.items()):
                if cell_1.is_lf:
                    cell_idx_1  = cell_idxs_1[cell_key_1]
                    ndof_th_1   = cell_1.ndofs[0]
                    cell_ndof_1 = ndof_x_1 * ndof_y_1 * ndof_th_1

                    cell_mtxs_01[cell_idx_0][cell_idx_1] = coo_matrix((cell_ndof_0, cell_ndof_1))
                    
    # Loop through cells of column C
    # For each cell in column C, we loop through the neighboring cells K^(n)
    # in neighboring column C'
    for cell_key_0, cell_0 in sorted(col_0.cells.items()):
        if cell_0.is_lf:
            # Get information about cell K in column C
            S_quad_0 = get_S_quad(cell_0)
            cell_idx_0 = cell_idxs_0[cell_key_0] # Matrix index of cell 0 in
                                               # column matrices
            [th0_0, th1_0]             = cell_0.pos
            dth_0                      = th1_0 - th0_0
            ndof_th_0                  = cell_0.ndofs[0]
            [_, _, _, _, thb_0, wth_0] = qd.quad_xyth(nnodes_th = ndof_th_0)
            
            # (p, q, r) => alpha
            # alpha is number of rows, always corresponds to K
            def alpha(pp, qq, rr):
                out = ndof_th_0 * ndof_y_0 * pp \
                    + ndof_th_0 * qq \
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
                    Theta_F = np.cos(thf_0)
                    dcoeff_0  = dy_0 * dth_0 / 4

                    ndof_0 = ndof_y_0 * ndof_th_0 # Number of *NON-ZERO* DoFs

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
                            Theta_F_a = Theta_F[aa]

                            alphalist[idx] = alpha(x_idx, jj, aa)
                            vlist[idx]  = dcoeff_0 * wy_0_j * wth_0_a * Theta_F_a

                            idx += 1
                    
                elif (F%2 == 1):
                    Theta_F = np.sin(thf_0)
                    dcoeff_0  = -dx_0 * dth_0 / 4

                    ndof_0    = ndof_x_0 * ndof_th_0 # Number of *NON-ZERO* DoFs

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
                            Theta_F_a = Theta_F[aa]

                            alphalist[idx] = alpha(ii, y_idx, aa)
                            vlist[idx]  = dcoeff_0 * wx_0_i * wth_0_a * Theta_F_a
                            
                            idx += 1

                cell_ndof_0 = ndof_x_0 * ndof_y_0 * ndof_th_0
                cell_mtxs_00[cell_idx_0] = coo_matrix((vlist, (alphalist, alphalist)),
                                                      shape = (cell_ndof_0, cell_ndof_0))
                
            else:
                # In Fm, we construct the inter-column cell matrices
                nhbr_cells = ji_mesh.get_cell_nhbr_in_col(cell_0, col_1)
                for cell_1 in nhbr_cells:
                    if cell_1:
                        if cell_1.is_lf:
                            cell_key_1 = cell_1.key
                            
                            cell_idx_1 = cell_idxs_1[cell_key_1]
                            
                            [th0_1, th1_1] = cell_1.pos
                            dth_1          = th1_1 - th0_1
                            ndof_th_1      = cell_1.ndofs[0]
                            [_, _, _, _, thb_1, wth_1] = qd.quad_xyth(nnodes_th =  ndof_th_1)
                            
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
                                    Theta_F = np.cos(thf_0)
                                    thb_0_1 = pull_back(th0_1, th1_1, thf_0)
                                    
                                    for aa in range(0, ndof_th_1):
                                        for rr in range(0, ndof_th_0):
                                            wth_0_r = wth_0[rr]
                                            Theta_F_r = Theta_F[rr]
                                            xi_a = qd.lag_eval(thb_1, aa, thb_0_1[rr])
                                            E_th[aa, rr] = wth_0_r * Theta_F_r * xi_a
                                            
                                elif ndof_th_0 < ndof_th_1:
                                    thf_1_0 = push_forward(th0_0, th1_0, thb_1)
                                    Theta_F = np.cos(thf_1_0)
                                    
                                    thb_1_0_1 = pull_back(th0_1, th1_1, thf_1_0)
                                    
                                    for aa in range(0, ndof_th_1):
                                        for rr in range(0, ndof_th_0):
                                            for aap in range(0, ndof_th_1):
                                                xi_a = qd.lag_eval(thb_1, aa, thb_1_0_1[aap])
                                                xi_r = qd.lag_eval(thb_0, rr, thb_1[aap])
                                                E_th[aa, rr] += wth_1[aap] * Theta_F[aap] * xi_a * xi_r
                                        
                                # Construct E^(K^(n)K,y)
                                E_y = np.zeros([ndof_y_1, ndof_y_0])
                                if ndof_y_0 >= ndof_y_1:
                                    yyf_0   = push_forward(y0_0, y1_0, yyb_0)
                                    yyb_0_1 = pull_back(y0_1, y1_1, yyf_0)
                                    
                                    for jj in range(0, ndof_y_1):
                                        for qq in range(0, ndof_y_0):
                                            wy_0_q = wy_0[qq]
                                            psi_j = qd.lag_eval(yyb_0_1, jj, yyb_0[qq])
                                            E_y[jj, qq] = wy_0_q * psi_j
                                            
                                elif ndof_y_0 < ndof_y_1:
                                    yyf_1_0   = push_forward(y0_0, y1_0, yyb_1)
                                    yyb_1_0_1 = pull_back(y0_1, y1_1, yyf_1_0)
                                    
                                    for jj in range(0, ndof_y_1):
                                        for qq in range(0, ndof_y_0):
                                            for jjp in range(0, ndof_y_1):
                                                psi_j = qd.lag_eval(yyb_1, jj, yyb_1_0_1[jjp])
                                                psi_q = qd.lag_eval(yyb_0, qq, yyb_1[jjp])
                                                E_y[jj, qq] += wy_1[jjp] * psi_j * psi_q
                                else:
                                    print_msg('ERROR - ndof_y_0 is neither <=, > ndof_y_1.')
                                    
                                dcoeff_0  = dy_0 * dth_0 / 4

                                # Number of *NON-ZERO* DoFs
                                ndof_0    = ndof_y_0 * ndof_th_0 
                                ndof_1    = ndof_y_1 * ndof_th_1
                                
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
                                    
                                idx = 0
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
                                    thb_0_1 = pull_back(th0_1, th1_1, thf_0)
                                    
                                    for aa in range(0, ndof_th_1):
                                        for rr in range(0, ndof_th_0):
                                            wth_0_r = wth_0[rr]
                                            Theta_F_r = Theta_F[rr]
                                            xi_a = qd.lag_eval(thb_1, aa,
                                                               thb_0_1[rr])
                                            E_th[aa, rr] = wth_0_r * Theta_F_r * xi_a
                                            
                                elif ndof_th_0 < ndof_th_1:
                                    thf_1_0 = push_forward(th0_0, th1_0, thb_1)
                                    Theta_F = np.cos(thf_1_0)
                                    
                                    thb_1_0_1 = pull_back(th0_1, th1_1, thf_1_0)
                                    
                                    for aa in range(0, ndof_th_1):
                                        for rr in range(0, ndof_th_0):
                                            for aap in range(0, ndof_th_1):
                                                xi_a = qd.lag_eval(thb_1, aa, thb_1_0_1[aap])
                                                xi_r = qd.lag_eval(thb_0, rr, thb_1[aap])
                                                E_th[aa, rr] += wth_1[aap] * Theta_F[aap] * xi_a * xi_r
                                                
                                # Construct E^(K^(n)K,x)
                                E_x = np.zeros([ndof_x_1, ndof_x_0])
                                if ndof_x_0 >= ndof_x_1:
                                    xxf_0   = push_forward(x0_0, x1_0, xxb_0)
                                    xxb_0_1 = pull_back(x0_1, x1_1, xxf_0)
                                
                                    for ii in range(0, ndof_x_1):
                                        for pp in range(0, ndof_x_0):
                                            wx_0_p = wx_0[pp]
                                            phi_i = qd.lag_eval(xxb_0_1, ii, xxb_0[pp])
                                            E_x[ii, pp] = wx_0_p * phi_i
                                            
                                elif ndof_x_0 < ndof_x_1:
                                    xxf_1_0   = push_forward(x0_0, x1_0, xb_1)
                                    xxb_1_0_1 = pull_back(x0_1, x1_1, xf_1_0)
                                    
                                    for ii in range(0, ndof_x_1):
                                        for pp in range(0, ndof_x_0):
                                            for iip in range(0, ndof_x_1):
                                                phi_i = qd.lag_eval(xb_1, jj, xxb_1_0_1[iip])
                                                phi_j = qd.lag_eval(xxb_0, qq, xxb_1[iip])
                                                E_x[jj, qq] += wx_1[iip] * phi_i * phi_p
                                            
                                dcoeff_0  = -dx_0 * dth_0 / 4
                                
                                # Number of *NON-ZERO* DoFs
                                ndof_0    = ndof_x_0 * ndof_th_0
                                ndof_1    = ndof_x_1 * ndof_th_1
                                
                                alphalist = np.zeros([ndof_0 * ndof_1], dtype = np.int32) # alpha index
                                betalist  = np.zeros([ndof_0 * ndof_1], dtype = np.int32) # beta index
                                vlist     = np.zeros([ndof_0 * ndof_1]) # Entry value
                                
                                if (F == 1):
                                    y_idx_0 = ndof_y_0 - 1
                                    y_idx_1 = 0
                                elif (F == 3):
                                    y_idx_0 = 0
                                    y_idx_1 = ndof_y_1 - 1
                                    
                                idx = 0
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

                        cell_ndof_0 = ndof_x_0 * ndof_y_0 * ndof_th_0
                        cell_ndof_1 = ndof_x_1 * ndof_y_1 * ndof_th_1

                        col_ndof_0 += cell_ndof_0
                        col_ndof_1 += cell_ndof_1
                        
                        cell_mtxs_00[cell_idx_0] = coo_matrix((cell_ndof_0, cell_ndof_0))
                        cell_mtxs_01[cell_idx_0][cell_idx_1] =\
                            coo_matrix((vlist, (alphalist, betalist)),
                                       shape = (cell_ndof_0, cell_ndof_1))
                        
    col_mtx_00 = block_diag(cell_mtxs_00, format = 'csr')
    col_mtx_01 = bmat(cell_mtxs_01, format = 'csr')


    return [col_mtx_00, col_mtx_01]
