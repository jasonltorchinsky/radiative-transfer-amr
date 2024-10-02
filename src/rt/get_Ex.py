import numpy as np
from copy import deepcopy

from dg.projection import push_forward, pull_back
from dg.quadrature import lag_eval, quad_xyth

Ex_matrices = {}


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

    # _0 <=> K in equations
    # _1 <=> K" in equations

    # Get the neighbor relation
    if lv_0 == lv_1:
        pos_str = "s"
    elif lv_0 - lv_1 == -1:
        if mid_1 < mid_0:
            pos_str = "l"
        else: # mid_0 < mid_1
            pos_str = "u"
    elif lv_0 - lv_1 == 1:
        if mid_1 < mid_0:
            pos_str = "u"
        else: # mid_0 < mid_1
            pos_str = "l"
        
    nhbr_rel = (lv_0 - lv_1, pos_str)
    
    key = (ndof_x_0, ndof_x_1, nhbr_rel)
    if key in Ex_matrices.keys():
        return Ex_matrices[key]
    
    [xxb_0, wx_0, _, _, _, _] = quad_xyth(nnodes_x = ndof_x_0)
    [xxb_1, wx_1, _, _, _, _] = quad_xyth(nnodes_x = ndof_x_1)
    E_x = np.zeros([ndof_x_1, ndof_x_0])
    
    # If _1 is more refined, then its basis functions aren"t supported on half
    # the interval, and we must integrate on that interval instead
    if lv_0 - lv_1 == -1:
        coeff = 1. / 2.
        if ndof_x_0 > ndof_x_1:
            phi_ip_matrix = np.zeros([ndof_x_1, ndof_x_0])
            for ii in range(0, ndof_x_1):
                for pp_p in range(0, ndof_x_0):
                    phi_ip_matrix[ii, pp_p] = lag_eval(xxb_1, ii, xxb_0[pp_p])
                    
            xxf_0_1 = push_forward(x0_1, x1_1, xxb_0)
            xxb_0_1_0 = pull_back(x0_0, x1_0, xxf_0_1)
            phi_pp_matrix = np.zeros([ndof_x_0, ndof_x_0])
            for pp in range(0, ndof_x_0):
                for pp_p in range(0, ndof_x_0):
                    phi_pp_matrix[pp, pp_p] = lag_eval(xxb_0, pp, xxb_0_1_0[pp_p])
                    
            for ii in range(0, ndof_x_1):
                for pp in range(0, ndof_x_0):
                    for pp_p in range(0, ndof_x_0):
                        wx_rp = wx_0[pp_p]
                        phi_ipp = phi_ip_matrix[ii, pp_p]
                        phi_ppp = phi_pp_matrix[pp, pp_p]
                        
                        E_x[ii, pp] += coeff * wx_rp * phi_ipp * phi_ppp
        else:
            xxf_1 = push_forward(x0_1, x1_1, xxb_1)            
            xxb_1_0 = pull_back(x0_0, x1_0, xxf_1)
            phi_pi_matrix = np.zeros([ndof_x_0, ndof_x_1])
            for pp in range(0, ndof_x_0):
                for ii in range(0, ndof_x_1):
                    phi_pi_matrix[pp, ii] = lag_eval(xxb_0, pp, xxb_1_0[ii])
                    
            for ii in range(0, ndof_x_1):
                wx_i = wx_1[ii]
                for pp in range(0, ndof_x_0):
                    phi_pi = phi_pi_matrix[pp, ii]
                    
                    E_x[ii, pp] = coeff * wx_i * phi_pi
    else:
        if ndof_x_0 >= ndof_x_1:
            xxf_0 = push_forward(x0_0, x1_0, xxb_0)
            xxb_0_1 = pull_back(x0_1, x1_1, xxf_0)
            phi_ip_matrix = np.zeros([ndof_x_1, ndof_x_0])
            for ii in range(0, ndof_x_1):
                for pp in range(0, ndof_x_0):
                    phi_ip_matrix[ii, pp] = lag_eval(xxb_1, ii, xxb_0_1[pp])
                    
            for ii in range(0, ndof_x_1):
                for pp in range(0, ndof_x_0):
                    wx_r = wx_0[pp]
                    phi_ip = phi_ip_matrix[ii, pp]
                    
                    E_x[ii, pp] = wx_r * phi_ip
        else:
            xxf_1_0 = push_forward(x0_0, x1_0, xxb_1)
            xxb_1_0_1 = pull_back(x0_1, x1_1, xxf_1_0)
            phi_ii_matrix = np.zeros([ndof_x_1, ndof_x_1])
            for ii in range(0, ndof_x_1):
                for ii_p in range(0, ndof_x_1):
                    phi_ii_matrix[ii, ii_p] = lag_eval(xxb_1, ii, xxb_1_0_1[ii_p])
                    
            phi_pi_matrix = np.zeros([ndof_x_0, ndof_x_1])
            for pp in range(0, ndof_x_0):
                for ii_p in range(0, ndof_x_1):
                    phi_pi_matrix[pp, ii_p] = lag_eval(xxb_0, pp, xxb_1[ii_p])
                    
            for ii in range(0, ndof_x_1):
                for pp in range(0, ndof_x_0):
                    for ii_p in range(0, ndof_x_1):
                        wx_ip = wx_1[ii_p]
                        phi_iip = phi_ii_matrix[ii, ii_p]
                        phi_pip = phi_pi_matrix[pp, ii_p]
                        
                        E_x[ii, pp] += wx_ip * phi_iip * phi_pip
                        
    Ex_matrices[key] = deepcopy(E_x)
    return E_x

def get_Ex_old(mesh, col_key_0, col_key_1):
    
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

    # _0 <=> K in equations
    # _1 <=> K" in equations

    # Get the neighbor relation
    if lv_0 == lv_1:
        pos_str = "s"
    elif lv_0 - lv_1 == -1:
        if mid_1 < mid_0:
            pos_str = "l"
        else: # mid_0 < mid_1
            pos_str = "u"
    elif lv_0 - lv_1 == 1:
        if mid_1 < mid_0:
            pos_str = "u"
        else: # mid_0 < mid_1
            pos_str = "l"
        
    nhbr_rel = (lv_0 - lv_1, pos_str)
    
    key = (ndof_x_0, ndof_x_1, nhbr_rel)
    if key in Ex_matrices.keys():
        return Ex_matrices[key]
    
    # Handles if K, K" are of different refinement levels
    if nhbr_rel[0] == -1:
        coeff = 0.5
        if nhbr_rel[1] == "l":
            def f(x):
                return 2. * x + 1.
            
            def f_inv(x):
                return 0.5 * (x - 1.)
        else: # if nhbr_rel[1] == "u"
            def f(x):
                return 2. * x - 1.
            
            def f_inv(x):
                return 0.5 * (x + 1.)
    else:
        coeff = 1.
        def f(x):
            return x

        def f_inv(x):
            return x
    
    if ndof_x_0 >= ndof_x_1:
        [xxb_0, wx_0, _, _, _, _] = quad_xyth(nnodes_x = ndof_x_0)
        [xxb_1, _, _, _, _, _]    = quad_xyth(nnodes_x = ndof_x_1)
        finv_xxb_0 = f_inv(xxb_0)
        
        phi_pp_matrix = np.zeros([ndof_x_0, ndof_x_0])
        for pp in range(0, ndof_x_0):
            for pp_p in range(0, ndof_x_0):
                phi_pp_matrix[pp, pp_p] = lag_eval(xxb_0, pp, finv_xxb_0[pp_p])
        
        phi_ip_matrix = np.zeros([ndof_x_1, ndof_x_0])
        finv_xxf_0 = push_forward(x0_0, x1_0, finv_xxb_0)
        finv_xxb_0_1 = pull_back(x0_1, x1_1, finv_xxf_0)
        for ii in range(0, ndof_x_1):
            for pp_p in range(0, ndof_x_0):
                phi_ip_matrix[ii, pp_p] = lag_eval(xxb_1, ii, finv_xxb_0_1[pp_p])
        
        E_x = np.zeros([ndof_x_1, ndof_x_0])
        for ii in range(0, ndof_x_1):
            for pp in range(0, ndof_x_0):
                for pp_p in range(0, ndof_x_0):
                    E_x[ii, pp] += coeff * wx_0[pp_p] \
                        * phi_ip_matrix[ii, pp_p] * phi_pp_matrix[pp, pp_p]
        
    else:
        [xxb_0, _, _, _, _, _]    = quad_xyth(nnodes_x = ndof_x_0)
        [xxb_1, wx_1, _, _, _, _] = quad_xyth(nnodes_x = ndof_x_1)
        finv_xxb_1 = f_inv(xxb_1)
        
        phi_pi_matrix = np.zeros([ndof_x_0, ndof_x_1])
        for pp in range(0, ndof_x_0):
            for ii_p in range(0, ndof_x_1):
                phi_pi_matrix[pp, ii_p] = lag_eval(xxb_0, pp, finv_xxb_1[ii_p])
        
        phi_ii_matrix = np.zeros([ndof_x_1, ndof_x_1])
        finv_xxf_1_0 = push_forward(x0_0, x1_0, finv_xxb_1)
        finv_xxb_1_0 = pull_back(x0_1, x1_1, finv_xxf_1_0)
        for ii in range(0, ndof_x_1):
            for ii_p in range(0, ndof_x_1):
                phi_ii_matrix[ii, ii_p] = lag_eval(xxb_1, ii, finv_xxb_1_0[ii_p])
        
        E_x = np.zeros([ndof_x_1, ndof_x_0])
        for ii in range(0, ndof_x_1):
            for pp in range(0, ndof_x_0):
                for ii_p in range(0, ndof_x_1):
                    E_x[ii, pp] += coeff * wx_1[ii_p] \
                        * phi_ii_matrix[ii, ii_p] * phi_pi_matrix[pp, ii_p]

    return E_x
