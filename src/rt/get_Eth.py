import numpy as np

from dg.projection import get_f2f_matrix, push_forward
from dg.quadrature import quad_xyth

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
    
    if ndof_th_0 > ndof_th_1:
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
        
        [_, _, _, _, thb_0, wth_0] = quad_xyth(nnodes_th = ndof_th_0)

        ## WORK ON THIS, MAKE SURE IT'S RIGHT
        if nhbr_rel[0] == -1: # Basis functions not continuous on whole interval,
            # Must do integral over half interval
            if nhbr_rel[1] == 'l':
                thb_0 = 0.5 * (thb_0 - 1.)
            else: # nhbr_rel[1] == 'u'
                thb_0 = 0.5 * (thb_0 + 1.)
        
        thf_0 = push_forward(th0_0, th1_0, thb_0).reshape([1, ndof_th_0])
        Theta_F = Theta_F_func(thf_0, F)
        
        wth_0 = wth_0.reshape([1, ndof_th_0])
        
        xsi_ar_matrix = get_f2f_matrix(dim_str  = 'th',
                                       nbasis   = ndof_th_1,
                                       nnode    = ndof_th_0,
                                       nhbr_rel = nhbr_rel
                                       )

        E_th = wth_0 * Theta_F * xsi_ar_matrix

    else:
        [_, _, _, _, thb_1, wth_1] = quad_xyth(nnodes_th = ndof_th_1)

        thf_1_0 = push_forward(th0_0, th1_0, thb_1)
        Theta_F = Theta_F_func(thf_1_0, F)
        
        xsi_ra_matrix = get_f2f_matrix(dim_str  = 'th',
                                       nbasis   = ndof_th_0,
                                       nnode    = ndof_th_1,
                                       nhbr_rel = (0, 's')
                                       )
        
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
        
        xsi_aa_matrix = get_f2f_matrix(dim_str  = 'th',
                                       nbasis   = ndof_th_1,
                                       nnode    = ndof_th_1,
                                       nhbr_rel = nhbr_rel
                                       )

        E_th = np.zeros([ndof_th_1, ndof_th_0])
        for aa in range(0, ndof_th_1):
            for rr in range(0, ndof_th_0):
                for aap in range(0, ndof_th_1):
                    E_th[aa, rr] += wth_1[aap] * Theta_F[aap] \
                        * xsi_aa_matrix[aa, aap] * xsi_ra_matrix[rr, aap]

    return E_th


# Theta^F function
def Theta_F_func(theta, F):
    return np.cos(theta - F * np.pi / 2.)
