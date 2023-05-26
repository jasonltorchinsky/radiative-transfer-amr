import numpy as np

from dg.projection import push_forward, pull_back
from dg.quadrature import lag_eval, quad_xyth

Eth_matrices = {}

def get_Eth(mesh, col_key_0, cell_key_0, col_key_1, cell_key_1, F):
    
    col_0  = mesh.cols[col_key_0]
    cell_0 = col_0.cells[cell_key_0]
    [ndof_th_0] = cell_0.ndofs[:]
    lv_0   = cell_0.lv
    [th0_0, th1_0] = cell_0.pos[:]
    mid_0  = (th0_0 + th1_0) / 2.
    
    col_1  = mesh.cols[col_key_1]
    cell_1 = col_1.cells[cell_key_1]
    [ndof_th_1] = cell_1.ndofs[:]
    lv_1   = cell_1.lv
    [th0_1, th1_1] = cell_1.pos[:]
    mid_1  = (th0_1 + th1_1) / 2.

    # Get the neighbor relation
    if lv_0 == lv_1:
        pos_str = 's'
    elif lv_0 - lv_1 == -1:
        if mid_1 < mid_0:
            pos_str = 'l'
        else: # mid_0 < mid_1
            pos_str = 'u'
    elif lv_0 - lv_1 == 1:
        if mid_1 < mid_0:
            pos_str = 'u'
        else: # mid_0 < mid_1
            pos_str = 'l'
    
    nhbr_rel = (lv_0 - lv_1, pos_str)

    key = (ndof_th_0, ndof_th_1, nhbr_rel)
    if key in Eth_matrices.keys():
        return Eth_matrices[key]
    
    # Handles if K, K' are of different refinement levels
    if nhbr_rel[0] == -1:
        coeff = 0.5
        if nhbr_rel[1] == 'l':
            def f(th):
                return 2. * th + 1.
            
            def f_inv(th):
                return 0.5 * (th - 1.)
        else: # if nhbr_rel[1] == 'u'
            def f(th):
                return 2. * th - 1.
            
            def f_inv(th):
                return 0.5 * (th + 1.)
    else:
        coeff = 1.
        def f(th):
            return th

        def f_inv(th):
            return th
    
    if ndof_th_0 >= ndof_th_1:
        [_, _, _, _, thb_0, wth_0] = quad_xyth(nnodes_th = ndof_th_0)
        [_, _, _, _, thb_1, _]     = quad_xyth(nnodes_th = ndof_th_1)
        finv_thb_0 = f_inv(thb_0)
        
        finv_thf_0 = push_forward(th0_0, th1_0, finv_thb_0)
        Theta_F = Theta_F_func(finv_thf_0, F)
        
        xsi_rr_matrix = np.zeros([ndof_th_0, ndof_th_0])
        for rr in range(0, ndof_th_0):
            for rr_p in range(0, ndof_th_0):
                xsi_rr_matrix[rr, rr_p] = lag_eval(thb_0, rr, finv_thb_0[rr_p])
        
        xsi_ar_matrix = np.zeros([ndof_th_1, ndof_th_0])
        finv_thb_0_1 = pull_back(th0_1, th1_1, finv_thf_0)
        for aa in range(0, ndof_th_1):
            for rr_p in range(0, ndof_th_0):
                xsi_ar_matrix[aa, rr_p] = lag_eval(thb_1, aa, finv_thb_0_1[rr_p])
        
        E_th = np.zeros([ndof_th_1, ndof_th_0])
        for aa in range(0, ndof_th_1):
            for rr in range(0, ndof_th_0):
                for rr_p in range(0, ndof_th_0):
                    E_th[aa, rr] += coeff * wth_0[rr_p] * Theta_F[rr_p] \
                        * xsi_ar_matrix[aa, rr_p] * xsi_rr_matrix[rr, rr_p]

    else:
        [_, _, _, _, thb_0, _]     = quad_xyth(nnodes_th = ndof_th_0)
        [_, _, _, _, thb_1, wth_1] = quad_xyth(nnodes_th = ndof_th_1)
        finv_thb_1 = f_inv(thb_1)
        
        finv_thf_1_0 = push_forward(th0_0, th1_0, finv_thb_1)
        Theta_F = Theta_F_func(finv_thf_1_0, F)
        
        xsi_ra_matrix = np.zeros([ndof_th_0, ndof_th_1])
        for rr in range(0, ndof_th_0):
            for aa_p in range(0, ndof_th_1):
                xsi_ra_matrix[rr, aa_p] = lag_eval(thb_0, rr, finv_thb_1[aa_p])
        
        xsi_aa_matrix = np.zeros([ndof_th_1, ndof_th_1])
        finv_thb_1_0_1 = pull_back(th0_1, th1_1, finv_thf_1_0)
        for aa in range(0, ndof_th_1):
            for aa_p in range(0, ndof_th_1):
                xsi_aa_matrix[aa, aa_p] = lag_eval(thb_1, aa, finv_thb_1_0_1[aa_p])
        
        E_th = np.zeros([ndof_th_1, ndof_th_0])
        for aa in range(0, ndof_th_1):
            for rr in range(0, ndof_th_0):
                for aa_p in range(0, ndof_th_1):
                    E_th[aa, rr] += coeff * wth_1[aa_p] * Theta_F[aa_p] \
                        * xsi_aa_matrix[aa, aa_p] * xsi_ra_matrix[rr, aa_p]

    return E_th


# Theta^F function
def Theta_F_func(theta, F):
    return np.cos(theta - F * np.pi / 2.)
