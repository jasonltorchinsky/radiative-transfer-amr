import numpy as np

from dg.projection import get_f2f_matrix
from dg.quadrature import quad_xyth

def get_Ex(mesh, col_key_0, col_key_1):
    
    col_0    = mesh.cols[col_key_0]
    ndof_x_0 = col_0.ndofs[0]
    lv_0     = col_0.lv
    [x0_0, _, x1_0, _] = col_0.pos[:]
    mid_0    = (x0_0 + x1_0) / 2.
    
    col_1    = mesh.cols[col_key_1]
    ndof_x_1 = col_1.ndofs[0]
    lv_1     = col_1.lv
    [x0_1, _, x1_1, _] = col_1.pos[:]
    mid_1    = (x0_1 + x1_1) / 2.
    
    if ndof_x_0 >= ndof_x_1:
        [_, wx_0, _, _, _, _] = quad_xyth(nnodes_x = ndof_x_0)
        wx_0 = wx_0.reshape(1, ndof_x_0)
        
        if lv_0 == lv_1:
            pos_str = 's'
        elif lv_1 - lv_0 == -1:
            if mid_1 > mid_0:
                pos_str = 'l'
            else: # mid_0 > mid_1
                pos_str = 'u'
        elif lv_1 - lv_0 == 1:
            if mid_1 > mid_0:
                pos_str = 'u'
            else: # mid_0 > mid_1
                pos_str = 'l'
        
        phi_ip_matrix = get_f2f_matrix(dim_str  = 'x',
                                       nbasis   = ndof_x_1,
                                       nnode    = ndof_x_0,
                                       nhbr_rel = (lv_1 - lv_0, pos_str)
                                       )

        E_x = wx_0 * phi_ip_matrix

    else:
        [_, wx_1, _, _, _, _] = quad_xyth(nnodes_x = ndof_x_1)
        
        phi_pi_matrix = get_f2f_matrix(dim_str  = 'x',
                                       nbasis   = ndof_x_0,
                                       nnode    = ndof_x_1,
                                       nhbr_rel = (0, 's')
                                       )
        
        if lv_0 == lv_1:
            pos_str = 's'
        elif lv_1 - lv_0 == -1:
            if mid_1 > mid_0:
                pos_str = 'l'
            else: # mid_0 > mid_1
                pos_str = 'u'
        elif lv_1 - lv_0 == 1:
            if mid_1 > mid_0:
                pos_str = 'u'
            else: # mid_0 > mid_1
                pos_str = 'l'
        
        phi_ii_matrix = get_f2f_matrix(dim_str  = 'x',
                                       nbasis   = ndof_x_1,
                                       nnode    = ndof_x_1,
                                       nhbr_rel = (lv_1 - lv_0, pos_str)
                                       )
        
        E_x = np.zeros([ndof_x_1, ndof_x_0])
        for ii in range(0, ndof_x_1):
            for pp in range(0, ndof_x_0):
                for iip in range(0, ndof_x_1):
                    E_x[ii, pp] += wx_1[iip] \
                        * phi_ii_matrix[ii, iip] * phi_pi_matrix[pp, iip]

    return E_x
