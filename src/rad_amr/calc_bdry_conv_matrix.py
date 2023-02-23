import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag, bmat

from .Projection import Projection_2D
from .matrix_utils import push_forward, pull_back, get_col_idxs, \
    get_cell_idxs, get_idx_map

from dg.mesh import ji_mesh
import dg.quadrature as qd

from utils import print_msg


def calc_bdry_conv_matrix(mesh):

    # Variables that are the same throughout the loops
    nhbr_locs = ['+', '-']
    col_items = sorted(mesh.cols.items())
    
    # Create column indexing for constructing global mass matrix
    [ncols, col_idxs] = get_col_idxs(mesh)
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
    for col_key_0, col_0 in col_items:
        if col_0.is_lf:
            # Use _0 to refer to column C
            # Later, use _1 to refer to column C'
            col_idx_0 = col_idxs[col_key_0]

            # Loop through the faces of C
            for F in range(0, 4):
                if col_0.bdry[F]: # If column is on the spatial domain boundary,
                    # no need to find a neightbor.
                    [col_mtx_00, col_mtx_01] = calc_col_matrix(col_0, col_0, F)
                    
                    # The intra-column matrix may already exist. If so, add to it.
                    if col_mtxs[col_idx_0][col_idx_0] is None:
                        col_mtxs[col_idx_0][col_idx_0] = col_mtx_00 + col_mtx_01
                    else:                        
                        col_mtxs[col_idx_0][col_idx_0] += col_mtx_00 + col_mtx_01
                        
                else: #not col.bdry[F], column is not on the domain boundary
                    axis = F%2
                    nhbr_loc = nhbr_locs[int(F/2)]
                    [_, col_nhbr_1, col_nhbr_2] = \
                        ji_mesh.get_col_nhbr(mesh, col_0,
                                             axis = axis,
                                             nhbr_loc = nhbr_loc)
                    
                    for col_1 in [col_nhbr_1, col_nhbr_2]:
                        if col_1:
                            if col_1.is_lf:
                                col_key_1 = col_1.key
                                col_idx_1 = col_idxs[col_key_1]
                                
                                [col_mtx_00, col_mtx_01] = calc_col_matrix(col_0,
                                                                           col_1,
                                                                           F)
                                
                                if col_mtxs[col_idx_0][col_idx_1] is None:
                                    col_mtxs[col_idx_0][col_idx_1] = col_mtx_01
                                else:
                                    col_mtxs[col_idx_0][col_idx_1] += col_mtx_01
                                    
                                if col_mtxs[col_idx_0][col_idx_0] is None:
                                    col_mtxs[col_idx_0][col_idx_0] = col_mtx_00
                                else:                        
                                    col_mtxs[col_idx_0][col_idx_0] += col_mtx_00
                                    
    # Global boundary convection matrix is not block-diagonal
    # but we arranged the column matrices in the proper form
    bdry_conv_mtx = bmat(col_mtxs, format = 'csr')

    return bdry_conv_mtx

def calc_col_matrix(col_0, col_1, F):
    """
    Create the column interaction matrix between col_0, col_1.
    """

    # Get information about column C
    # _0 => Cell K in equations (in column C)
    # **b => Pull back coordinates (in [-1, 1])
    # w* => Quadrature weights
    cell_items_0            = sorted(col_0.cells.items())
    [ndof_x_0, ndof_y_0]    = col_0.ndofs
    [ncells_0, cell_idxs_0] = get_cell_idxs(col_0)
    cell_mtxs_00            = [None] * ncells_0

    # Set array to store cell matrices for inter-column matrices
    # Certainly not block diagonal, but set sizes later once we get number of
    # cells in neighboring columns
    cell_items_1            = sorted(col_1.cells.items())
    [ndof_x_1, ndof_y_1]    = col_1.ndofs
    [ncells_1, cell_idxs_1] = get_cell_idxs(col_1)
    cell_mtxs_01            = [[None] * ncells_1 for K in range(0, ncells_0)]

    # To ensure proper matrix construction, we initialize all cell
    # matrices to be empty sparse matrices
    for cell_key_0, cell_0 in cell_items_0:
        if cell_0.is_lf:
            cell_idx_0  = cell_idxs_0[cell_key_0]
            [ndof_th_0] = cell_0.ndofs
            cell_ndof_0 = ndof_x_0 * ndof_y_0 * ndof_th_0

            cell_mtxs_00[cell_idx_0] = \
                coo_matrix((cell_ndof_0, cell_ndof_0))
            
            for cell_key_1, cell_1 in cell_items_1:
                if cell_1.is_lf:
                    cell_idx_1  = cell_idxs_1[cell_key_1]
                    [ndof_th_1] = cell_1.ndofs
                    cell_ndof_1 = ndof_x_1 * ndof_y_1 * ndof_th_1

                    cell_mtxs_01[cell_idx_0][cell_idx_1] = \
                        coo_matrix((cell_ndof_0, cell_ndof_1))
                    
    # Loop through cells of column C
    # For each cell in column C, we loop through the neighboring cells K^(n)
    # in neighboring column C'
    for cell_key_0, cell_0 in cell_items_0:
        if cell_0.is_lf:
            # Get information about cell K in column C
            S_quad_0   = cell_0.quad
            cell_idx_0 = cell_idxs_0[cell_key_0] # Matrix index of cell 0 in
                                                 # column matrices
                                                 
            # If a cell is in F^+, contribute to column-matrix M^CC.
            is_Fp = ( ((S_quad_0 == 0) and (F == 0 or F == 1)) or
                      ((S_quad_0 == 1) and (F == 1 or F == 2)) or
                      ((S_quad_0 == 2) and (F == 2 or F == 3)) or
                      ((S_quad_0 == 3) and (F == 3 or F == 0)) )

            # If we're in Fp we contribute to M^CC and use the first formula
            # Otherwise we have the option of using the quadrature rule from
            # the neighboring column/cell
            if is_Fp:
                cell_mtxs_00[cell_idx_0] += \
                    calc_cell_matrix(cell_0, col_0, cell_0, col_0, F)
            else:
                nhbr_cells = ji_mesh.get_cell_nhbr_in_col(cell_0, col_1)
                for cell_1 in nhbr_cells:
                    if cell_1:
                        if cell_1.is_lf:
                            cell_key_1 = cell_1.key
                            cell_idx_1 = cell_idxs_1[cell_key_1]

                            cell_mtxs_01[cell_idx_0][cell_idx_1] += \
                                calc_cell_matrix(cell_0, col_0, cell_1, col_1, F)
                    
    col_mtx_00 = block_diag(cell_mtxs_00, format = 'csr')
    col_mtx_01 = bmat(      cell_mtxs_01, format = 'csr')

    return [col_mtx_00, col_mtx_01]

def calc_cell_matrix(cell_0, col_0, cell_1, col_1, F):
    """
    Create the column interaction matrix between col_0, col_1.
    """

    S_quad_0   = cell_0.quad
    
    # If a cell is in F^+, contribute to column-matrix M^CC.
    is_Fp = ( ((S_quad_0 == 0) and (F == 0 or F == 1)) or
              ((S_quad_0 == 1) and (F == 1 or F == 2)) or
              ((S_quad_0 == 2) and (F == 2 or F == 3)) or
              ((S_quad_0 == 3) and (F == 3 or F == 0)) )

    # Get information about column C
    # _0 => Cell K in equations (in column C)
    # **b => Pull back coordinates (in [-1, 1])
    # w* => Quadrature weights
    [x0_0, y0_0, x1_0, y1_0]         = col_0.pos
    [dx_0, dy_0]                     = [x1_0 - x0_0, y1_0 - y0_0]
    [ndof_x_0, ndof_y_0]             = col_0.ndofs
    [xxb_0, wx_0, yyb_0, wy_0, _, _] = qd.quad_xyth(nnodes_x = ndof_x_0,
                                                    nnodes_y = ndof_y_0)
    [th0_0, th1_0] = cell_0.pos
    dth_0          = th1_0 - th0_0
    [ndof_th_0]    = cell_0.ndofs
    [xxb_0, wx_0, yyb_0, wy_0, thb_0, wth_0] = \
        qd.quad_xyth(nnodes_x  = ndof_x_0,
                     nnodes_y  = ndof_y_0,
                     nnodes_th = ndof_th_0)
    
    cell_ndof_0    = ndof_x_0 * ndof_y_0 * ndof_th_0
    
    
    # Set array to store cell matrices for inter-column matrices
    # Certainly not block diagonal, but set sizes later once we get number of
    # cells in neighboring columns
    [x0_1, y0_1, x1_1, y1_1]         = col_1.pos
    [dx_1, dy_1]                     = [x1_1 - x0_1, y1_1 - y0_1]
    [ndof_x_1, ndof_y_1]             = col_1.ndofs
    
    [th0_1, th1_1] = cell_1.pos
    dth_1          = th1_1 - th0_1
    [ndof_th_1]    = cell_1.ndofs
    
    [xxb_1, wx_1, yyb_1, wy_1, thb_1, wth_1] = \
        qd.quad_xyth(nnodes_x  = ndof_x_1,
                     nnodes_y  = ndof_y_1,
                     nnodes_th = ndof_th_1)
    
    cell_ndof_1    = ndof_x_1 * ndof_y_1 * ndof_th_1

    # If on the inflow boundary of the domain, return a zero matrix
    if col_0.bdry[F] and not is_Fp:
        cell_mtx = coo_matrix((cell_ndof_0, cell_ndof_1))
        return cell_mtx

    # Theta^F function
    def Theta_F_func(theta):
        return np.cos(theta - F * np.pi / 2)

    # Set up non-Delta theta part of dcoeff
    if (F == 0) or (F == 2):
        dcoeff = (dy_0 * dth_0) / 4.
    else:
        dcoeff = (dx_0 * dth_0) / 4.
        

    # alpha is number of rows, beta is the number of columns
    alpha = get_idx_map(ndof_x_0, ndof_y_0, ndof_th_0)
    beta  = get_idx_map(ndof_x_1, ndof_y_1, ndof_th_1)
                    
    if (F%2 == 0): # Construct E^K'K,y_jq
        E_y = np.zeros([ndof_y_1, ndof_y_0])
        if ndof_y_0 >= ndof_y_1:
            yyf_0   = push_forward(y0_0, y1_0, yyb_0)
            yyb_0_1 = pull_back(   y0_1, y1_1, yyf_0)
            
            for jj in range(0, ndof_y_1):
                for qq in range(0, ndof_y_0):
                    wy_0_q = wy_0[qq]
                    psi_j  = qd.lag_eval(yyb_1, jj, yyb_0_1[qq])
                    E_y[jj, qq] = wy_0_q * psi_j
                    
        else: # ndof_y_0 < ndof_y_1
            yyf_1_0   = push_forward(y0_0, y1_0, yyb_1  )
            yyb_1_0_1 = pull_back(   y0_1, y1_1, yyf_1_0)
            
            for jj in range(0, ndof_y_1):
                for qq in range(0, ndof_y_0):
                    for jjp in range(0, ndof_y_1):
                        psi_j = qd.lag_eval(yyb_1, jj, yyb_1_0_1[jjp])
                        psi_q = qd.lag_eval(yyb_0, qq, yyb_1[jjp]    )
                        E_y[jj, qq] += wy_1[jjp] * psi_j * psi_q
                                
    else: # F%2 == 1, construct E^K'K,x_ip
        E_x = np.zeros([ndof_x_1, ndof_x_0])
        if ndof_x_0 >= ndof_x_1:
            xxf_0   = push_forward(x0_0, x1_0, xxb_0)
            xxb_0_1 = pull_back(   x0_1, x1_1, xxf_0)
            
            for ii in range(0, ndof_x_1):
                for pp in range(0, ndof_x_0):
                    wx_0_p = wx_0[pp]
                    phi_i  = qd.lag_eval(xxb_1, ii, xxb_0_1[pp])
                    E_x[ii, pp] = wx_0_p * phi_i
                    
        else: # if ndof_x_0 < ndof_x_1
            xxf_1_0   = push_forward(x0_0, x1_0, xxb_1  )
            xxb_1_0_1 = pull_back(   x0_1, x1_1, xxf_1_0)
            
            for ii in range(0, ndof_x_1):
                for pp in range(0, ndof_x_0):
                    for iip in range(0, ndof_x_1):
                        phi_i = qd.lag_eval(xxb_1, ii, xxb_1_0_1[iip])
                        phi_p = qd.lag_eval(xxb_0, pp, xxb_1[iip]    )
                        E_x[ii, pp] += wx_1[iip] * phi_i * phi_p
                        
    E_th = np.zeros([ndof_th_1, ndof_th_0])
    if ndof_th_0 >= ndof_th_1:
        thf_0   = push_forward(th0_0, th1_0, thb_0)
        Theta_F = Theta_F_func(thf_0)
        
        thb_0_1 = pull_back(th0_1, th1_1, thf_0)
        
        for aa in range(0, ndof_th_1):
            for rr in range(0, ndof_th_0):
                wth_0_r   = wth_0[rr]
                Theta_F_r = Theta_F[rr]
                xi_a      = qd.lag_eval(thb_1, aa, thb_0_1[rr])
                E_th[aa, rr] = wth_0_r * Theta_F_r * xi_a
                
    else: # ndof_th_0 < ndof_th_1
        thf_1_0 = push_forward(th0_0, th1_0, thb_1)
        Theta_F = Theta_F_func(thf_1_0)
        
        thb_1_0_1 = pull_back(th0_1, th1_1, thf_1_0)
        
        for aa in range(0, ndof_th_1):
            for rr in range(0, ndof_th_0):
                for aap in range(0, ndof_th_1):
                    xi_a = qd.lag_eval(thb_1, aa, thb_1_0_1[aap])
                    xi_r = qd.lag_eval(thb_0, rr, thb_1[aap]    )
                    E_th[aa, rr] += wth_1[aap] * Theta_F[aap] * xi_a * xi_r
                    
    # Many quantities are actually dependent only on
    # the parity of F not F itself, so we can 
    # separate primarily into two cases
    
    
    if (F%2 == 0):
        # Number of *NON-ZERO* DoFs
        ndof_0 = ndof_y_0 * ndof_th_0 
        ndof_1 = ndof_y_1 * ndof_th_1
        
        # The entry locations are mostly dependent
        # on only the parity of F, and their value
        # is entirely dependent on F, so we can
        # handle those here, too
        alphalist = np.zeros([ndof_0 * ndof_1], dtype = np.int32)
        betalist  = np.zeros([ndof_0 * ndof_1], dtype = np.int32)
        vlist     = np.zeros([ndof_0 * ndof_1])
        
        # The x index depends on whether col_0 is actually
        # on the boundary of the domain or not.
        if col_0.bdry[F] or is_Fp: # On boundary of domain
            if (F == 0):
                x_idx_0 = ndof_x_0 - 1
            elif (F == 2):
                x_idx_0 = 0
            x_idx_1 = x_idx_0
        else:
            if (F == 0):
                x_idx_0 = ndof_x_0 - 1
                x_idx_1 = 0
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
                        vlist[idx]     = dcoeff * E_th_ar * E_y_jq
                        
                        idx += 1
                        
    elif (F%2 == 1):                            
        # Number of *NON-ZERO* DoFs
        ndof_0  = ndof_x_0 * ndof_th_0
        ndof_1  = ndof_x_1 * ndof_th_1
        
        alphalist = np.zeros([ndof_0 * ndof_1], dtype = np.int32)
        betalist  = np.zeros([ndof_0 * ndof_1], dtype = np.int32)
        vlist     = np.zeros([ndof_0 * ndof_1])
        
        # The y index depends on whether col_0 is actually
        # on the boundary of the domain or not.
        if col_0.bdry[F] or is_Fp: # On boundary of domain
            if (F == 1):
                y_idx_0 = ndof_y_0 - 1
            elif (F == 3):
                y_idx_0 = 0
            y_idx_1 = y_idx_0
        else:
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
                        vlist[idx] = dcoeff * E_th_ar * E_x_ip
                        
                        idx += 1
                        
    cell_mtx = coo_matrix((vlist, (alphalist, betalist)),
                          shape = (cell_ndof_0, cell_ndof_1))

    return cell_mtx
