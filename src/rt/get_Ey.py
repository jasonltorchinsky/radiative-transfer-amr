from copy import deepcopy
import numpy as np

from dg.projection import push_forward, pull_back, get_f2f_matrix
from dg.quadrature import lag_eval, quad_xyth

Ey_matrices = {}

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

    # Check the comments for get-f2f_matrix for which matrix is which.
    # We get the basis functions from _1 and the nodes from _0.
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
    
    key = (ndof_y_0, ndof_y_1, nhbr_rel)
    if key in Ey_matrices.keys():
        return Ey_matrices[key]

    [_, _, yyb_0, wy_0, _, _] = quad_xyth(nnodes_y = ndof_y_0)
    [_, _, yyb_1, wy_1, _, _] = quad_xyth(nnodes_y = ndof_y_1)
    E_y = np.zeros([ndof_y_1, ndof_y_0])
    
    # If _1 is more refined, then its basis functions aren't suqqorted on half
    # the interval, and we must integrate on that interval instead
    if lv_0 - lv_1 == -1:
        coeff = 1. / 2.
        if ndof_y_0 > ndof_y_1:
            psi_jq_matrix = np.zeros([ndof_y_1, ndof_y_0])
            for jj in range(0, ndof_y_1):
                for qq_p in range(0, ndof_y_0):
                    psi_jq_matrix[jj, qq_p] = lag_eval(yyb_1, jj, yyb_0[qq_p])
                    
            yyf_0_1 = push_forward(y0_1, y1_1, yyb_0)
            yyb_0_1_0 = pull_back(y0_0, y1_0, yyf_0_1)
            psi_qq_matrix = np.zeros([ndof_y_0, ndof_y_0])
            for qq in range(0, ndof_y_0):
                for qq_p in range(0, ndof_y_0):
                    psi_qq_matrix[qq, qq_p] = lag_eval(yyb_0, qq, yyb_0_1_0[qq_p])
                    
            for jj in range(0, ndof_y_1):
                for qq in range(0, ndof_y_0):
                    for qq_p in range(0, ndof_y_0):
                        wy_rp = wy_0[qq_p]
                        psi_jqq = psi_jq_matrix[jj, qq_p]
                        psi_qqp = psi_qq_matrix[qq, qq_p]
                        
                        E_y[jj, qq] += coeff * wy_rp * psi_jqq * psi_qqp
        else:
            yyf_1 = push_forward(y0_1, y1_1, yyb_1)            
            yyb_1_0 = pull_back(y0_0, y1_0, yyf_1)
            psi_qj_matrix = np.zeros([ndof_y_0, ndof_y_1])
            for qq in range(0, ndof_y_0):
                for jj in range(0, ndof_y_1):
                    psi_qj_matrix[qq, jj] = lag_eval(yyb_0, qq, yyb_1_0[jj])
                    
            for jj in range(0, ndof_y_1):
                wy_i = wy_1[jj]
                for qq in range(0, ndof_y_0):
                    psi_qj = psi_qj_matrix[qq, jj]
                    
                    E_y[jj, qq] = coeff * wy_i * psi_qj
    else:
        if ndof_y_0 >= ndof_y_1:
            yyf_0 = push_forward(y0_0, y1_0, yyb_0)
            yyb_0_1 = pull_back(y0_1, y1_1, yyf_0)
            psi_jq_matrix = np.zeros([ndof_y_1, ndof_y_0])
            for jj in range(0, ndof_y_1):
                for qq in range(0, ndof_y_0):
                    psi_jq_matrix[jj, qq] = lag_eval(yyb_1, jj, yyb_0_1[qq])
                    
            for jj in range(0, ndof_y_1):
                for qq in range(0, ndof_y_0):
                    wy_r = wy_0[qq]
                    psi_jq = psi_jq_matrix[jj, qq]
                    
                    E_y[jj, qq] = wy_r * psi_jq
        else:
            yyf_1_0 = push_forward(y0_0, y1_0, yyb_1)
            yyb_1_0_1 = pull_back(y0_1, y1_1, yyf_1_0)
            psi_jj_matrix = np.zeros([ndof_y_1, ndof_y_1])
            for jj in range(0, ndof_y_1):
                for jj_p in range(0, ndof_y_1):
                    psi_jj_matrix[jj, jj_p] = lag_eval(yyb_1, jj, yyb_1_0_1[jj_p])
                    
            psi_qj_matrix = np.zeros([ndof_y_0, ndof_y_1])
            for qq in range(0, ndof_y_0):
                for jj_p in range(0, ndof_y_1):
                    psi_qj_matrix[qq, jj_p] = lag_eval(yyb_0, qq, yyb_1[jj_p])
                    
            for jj in range(0, ndof_y_1):
                for qq in range(0, ndof_y_0):
                    for jj_p in range(0, ndof_y_1):
                        wy_jq = wy_1[jj_p]
                        psi_jjp = psi_jj_matrix[jj, jj_p]
                        psi_qjp = psi_qj_matrix[qq, jj_p]
                        
                        E_y[jj, qq] += wy_jq * psi_jjp * psi_qjp
                        
    Ey_matrices[key] = deepcopy(E_y)
    return E_y

def get_Ey_old(mesh, col_key_0, col_key_1):
    
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

    # Check the comments for get-f2f_matrix for which matrix is which.
    # We get the basis functions from _1 and the nodes from _0.
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
    
    key = (ndof_y_0, ndof_y_1, nhbr_rel)
    if key in Ey_matrices.keys():
        return Ey_matrices[key]

    # Handles if K, K' are of different refinement levels
    if nhbr_rel[0] == -1:
        coeff = 0.5
        if nhbr_rel[1] == 'l':
            def f(y):
                return 2. * y + 1.
            
            def f_inv(y):
                return 0.5 * (y - 1.)
        else: # if nhbr_rel[1] == 'u'
            def f(y):
                return 2. * y - 1.
            
            def f_inv(y):
                return 0.5 * (y + 1.)
    else:
        coeff = 1.
        def f(y):
            return y

        def f_inv(y):
            return y
    
    if ndof_y_0 >= ndof_y_1:
        [_, _, yyb_0, wy_0, _, _] = quad_xyth(nnodes_y = ndof_y_0)
        [_, _, yyb_1, _, _, _]    = quad_xyth(nnodes_y = ndof_y_1)
        finv_yyb_0 = f_inv(yyb_0)
        
        psi_qq_matrix = np.zeros([ndof_y_0, ndof_y_0])
        for qq in range(0, ndof_y_0):
            for qq_p in range(0, ndof_y_0):
                psi_qq_matrix[qq, qq_p] = lag_eval(yyb_0, qq, finv_yyb_0[qq_p])
        
        psi_jq_matrix = np.zeros([ndof_y_1, ndof_y_0])
        finv_yyf_0 = push_forward(y0_0, y1_0, finv_yyb_0)
        finv_yyb_0_1 = pull_back(y0_1, y1_1, finv_yyf_0)
        for jj in range(0, ndof_y_1):
            for qq_p in range(0, ndof_y_0):
                psi_jq_matrix[jj, qq_p] = lag_eval(yyb_1, jj, finv_yyb_0_1[qq_p])
        
        E_y = np.zeros([ndof_y_1, ndof_y_0])
        for jj in range(0, ndof_y_1):
            for qq in range(0, ndof_y_0):
                for qq_p in range(0, ndof_y_0):
                    E_y[jj, qq] += coeff * wy_0[qq_p] \
                        * psi_jq_matrix[jj, qq_p] * psi_qq_matrix[qq, qq_p]
        
    else:
        [_, _, yyb_0, _, _, _]    = quad_xyth(nnodes_y = ndof_y_0)
        [_, _, yyb_1, wy_1, _, _] = quad_xyth(nnodes_y = ndof_y_1)
        finv_yyb_1 = f_inv(yyb_1)
        
        psi_qj_matrix = np.zeros([ndof_y_0, ndof_y_1])
        for qq in range(0, ndof_y_0):
            for jj_p in range(0, ndof_y_1):
                psi_qj_matrix[qq, jj_p] = lag_eval(yyb_0, qq, finv_yyb_1[jj_p])
        
        psi_jj_matrix = np.zeros([ndof_y_1, ndof_y_1])
        finv_yyf_1_0 = push_forward(y0_0, y1_0, finv_yyb_1)
        finv_yyb_1_0 = pull_back(y0_1, y1_1, finv_yyf_1_0)
        for jj in range(0, ndof_y_1):
            for jj_p in range(0, ndof_y_1):
                psi_jj_matrix[jj, jj_p] = lag_eval(yyb_1, jj, finv_yyb_1_0[jj_p])
        
        E_y = np.zeros([ndof_y_1, ndof_y_0])
        for jj in range(0, ndof_y_1):
            for qq in range(0, ndof_y_0):
                for jj_p in range(0, ndof_y_1):
                    E_y[jj, qq] += coeff * wy_1[jj_p] \
                        * psi_jj_matrix[jj, jj_p] * psi_qj_matrix[qq, jj_p]
                    
    return E_y
