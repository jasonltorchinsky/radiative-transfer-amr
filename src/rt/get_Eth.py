from copy import deepcopy
import numpy as np

from dg.projection import push_forward, pull_back
from dg.quadrature import lag_eval, quad_xyth

#Eth_matrices = {}

def get_Eth(mesh, col_key_0, cell_key_0, col_key_1, cell_key_1, F):
    
    col_0          = mesh.cols[col_key_0]
    cell_0         = col_0.cells[cell_key_0]
    [ndof_th_0]    = cell_0.ndofs[:]
    lv_0           = cell_0.lv
    [th0_0, th1_0] = cell_0.pos[:]
    dth_0          = th1_0 - th0_0
    mid_0          = (th0_0 + th1_0) / 2.
    
    col_1          = mesh.cols[col_key_1]
    cell_1         = col_1.cells[cell_key_1]
    [ndof_th_1]    = cell_1.ndofs[:]
    lv_1           = cell_1.lv
    [th0_1, th1_1] = cell_1.pos[:]
    dth_1          = th1_1 - th0_1
    mid_1          = (th0_1 + th1_1) / 2.
    
    # _0 <=> K in equations
    # _1 <=> K' in equations
    
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
    # ISSUE: Theta_F is dependent on cell position, so we can't just reuse data
    #if key in Eth_matrices.keys():
    #    return Eth_matrices[key]
    
    [_, _, _, _, thb_0, wth_0] = quad_xyth(nnodes_th = ndof_th_0)
    [_, _, _, _, thb_1, wth_1] = quad_xyth(nnodes_th = ndof_th_1)
    E_th = np.zeros([ndof_th_1, ndof_th_0])
    
    # If _1 is more refined, then its basis functions aren't supported on half
    # the interval, and we must integrate on that interval instead
    if lv_0 - lv_1 == -1:
        coeff = 1. / 2.
        if ndof_th_0 > ndof_th_1:
            thf_0_1 = push_forward(th0_1, th1_1, thb_0)
            Theta_F = Theta_F_func(thf_0_1, F)
            
            xsi_ar_matrix = np.zeros([ndof_th_1, ndof_th_0])
            for aa in range(0, ndof_th_1):
                for rr_p in range(0, ndof_th_0):
                    xsi_ar_matrix[aa, rr_p] = lag_eval(thb_1, aa, thb_0[rr_p])
                    
            thb_0_1_0 = pull_back(th0_0, th1_0, thf_0_1)
            xsi_rr_matrix = np.zeros([ndof_th_0, ndof_th_0])
            for rr in range(0, ndof_th_0):
                for rr_p in range(0, ndof_th_0):
                    xsi_rr_matrix[rr, rr_p] = lag_eval(thb_0, rr, thb_0_1_0[rr_p])
                    
            for aa in range(0, ndof_th_1):
                for rr in range(0, ndof_th_0):
                    for rr_p in range(0, ndof_th_0):
                        wth_rp = wth_0[rr_p]
                        Theta_F_rp = Theta_F[rr_p]
                        xsi_arp = xsi_ar_matrix[aa, rr_p]
                        xsi_rrp = xsi_rr_matrix[rr, rr_p]
                        
                        E_th[aa, rr] += wth_rp * Theta_F_rp * xsi_arp * xsi_rrp
                        
        else: # ndof_th_1 >= ndof_th_0
            thf_1 = push_forward(th0_1, th1_1, thb_1)
            Theta_F = Theta_F_func(thf_1, F)
            
            thb_1_0 = pull_back(th0_0, th1_0, thf_1)
            xsi_ra_matrix = np.zeros([ndof_th_0, ndof_th_1])
            for rr in range(0, ndof_th_0):
                for aa in range(0, ndof_th_1):
                    xsi_ra_matrix[rr, aa] = lag_eval(thb_0, rr, thb_1_0[aa])
                    
            for aa in range(0, ndof_th_1):
                wth_a = wth_1[aa]
                Theta_F_a = Theta_F[aa]
                for rr in range(0, ndof_th_0):
                    xsi_ra = xsi_ra_matrix[rr, aa]
                    
                    E_th[aa, rr] = wth_a * Theta_F_a * xsi_ra
                    
        E_th *= coeff
        
    else:
        if ndof_th_0 >= ndof_th_1:
            thf_0 = push_forward(th0_0, th1_0, thb_0)
            Theta_F = Theta_F_func(thf_0, F)
            
            thb_0_1 = pull_back(th0_1, th1_1, thf_0)
            xsi_ar_matrix = np.zeros([ndof_th_1, ndof_th_0])
            for aa in range(0, ndof_th_1):
                for rr in range(0, ndof_th_0):
                    xsi_ar_matrix[aa, rr] = lag_eval(thb_1, aa, thb_0_1[rr])
                    
            for aa in range(0, ndof_th_1):
                for rr in range(0, ndof_th_0):
                    wth_r = wth_0[rr]
                    Theta_F_r = Theta_F[rr]
                    xsi_ar = xsi_ar_matrix[aa, rr]
                    
                    E_th[aa, rr] = wth_r * Theta_F_r * xsi_ar
                    
        else:
            thf_1_0 = push_forward(th0_0, th1_0, thb_1)
            Theta_F = Theta_F_func(thf_1_0, F)
            
            thb_1_0_1 = pull_back(th0_1, th1_1, thf_1_0)
            xsi_aa_matrix = np.zeros([ndof_th_1, ndof_th_1])
            for aa in range(0, ndof_th_1):
                for aa_p in range(0, ndof_th_1):
                    xsi_aa_matrix[aa, aa_p] = lag_eval(thb_1, aa, thb_1_0_1[aa_p])
                    
            xsi_ra_matrix = np.zeros([ndof_th_0, ndof_th_1])
            for rr in range(0, ndof_th_0):
                for aa_p in range(0, ndof_th_1):
                    xsi_ra_matrix[rr, aa_p] = lag_eval(thb_0, rr, thb_1[aa_p])
                    
            for aa in range(0, ndof_th_1):
                for rr in range(0, ndof_th_0):
                    for aa_p in range(0, ndof_th_1):
                        wth_ap = wth_1[aa_p]
                        Theta_F_ap = Theta_F[aa_p]
                        xsi_aap = xsi_aa_matrix[aa, aa_p]
                        xsi_rap = xsi_ra_matrix[rr, aa_p]
                        
                        E_th[aa, rr] += wth_ap * Theta_F_ap * xsi_aap * xsi_rap
                        
    #Eth_matrices[key] = deepcopy(E_th)
    E_th[np.abs(E_th) < 1.e-15] = 0.0
    return E_th

# Theta^F function
def Theta_F_func(theta, F):
    return np.cos(theta - F * np.pi / 2.)
