import numpy as np

from dg.projection import get_f2f_matrix
from dg.quadrature import quad_xyth

def get_Ey(mesh, col_key_0, col_key_1):
    
    col_0    = mesh.cols[col_key_0]
    ndof_y_0 = col_0.ndofs[1]
    lv_0     = col_0.lv
    [_, y0_0, _, y1_0] = col_0.pos[:]
    mid_0    = (y0_0 + y1_0) / 2.
    
    col_1    = mesh.cols[col_key_1]
    ndof_y_1 = col_1.ndofs[1]
    lv_1     = col_1.lv
    [_, y0_1, _, y1_1] = col_1.pos[:]
    mid_1    = (y0_1 + y1_1) / 2.
    
    if ndof_y_0 > ndof_y_1:
        [_, _, _, wy_0, _, _] = quad_xyth(nnodes_y = ndof_y_0)
        wy_0 = wy_0.reshape([1, ndof_y_0])
        
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

        nhbr_rel = (lv_1 - lv_0, pos_str)
                
        psi_jq_matrix = get_f2f_matrix(dim_str  = 'y',
                                       nbasis   = ndof_y_1,
                                       nnode    = ndof_y_0,
                                       nhbr_rel = nhbr_rel
                                       )
        
        E_y = wy_0 * psi_jq_matrix

    else:
        [_, _, _, wy_1, _, _] = quad_xyth(nnodes_y = ndof_y_1)
        
        psi_qj_matrix = get_f2f_matrix(dim_str  = 'y',
                                       nbasis   = ndof_y_0,
                                       nnode    = ndof_y_1,
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
                
        psi_jj_matrix = get_f2f_matrix(dim_str  = 'y',
                                       nbasis   = ndof_y_1,
                                       nnode    = ndof_y_1,
                                       nhbr_rel = nhbr_rel
                                       )

        E_y = np.zeros([ndof_y_1, ndof_y_0])
        for jj in range(0, ndof_y_1):
            for qq in range(0, ndof_y_0):
                for jjp in range(0, ndof_y_1):
                    E_y[jj, qq] += wy_1[jjp] \
                        * psi_jj_matrix[jj, jjp] * psi_qj_matrix[qq, jjp]

    return E_y
