import numpy as np

from dg.projection import push_forward, pull_back
from dg.quadrature import lag_eval, quad_xyth

def get_Eth(mesh, col_key_0, cell_key_0, col_key_1, cell_key_1, F):
    """
    Angular factor of the inflow term on face F of column C (_0) from the
    neighboring cell K" (_1) in column C" (_1).

    E_th[aa, rr] approximates the integral of Theta_F * xsi"_aa * xsi_rr over
    the overlap of K and K". It is collocated at the Legendre-Gauss nodes of K,
    the same rule used for the volume and outflow terms of K, so that all
    angular integrals of K share one quadrature and a constant state is
    preserved on non-conforming angular meshes. The neighbor"s basis function
    is evaluated at the nodes of K that lie in K". A node of K on the boundary
    between two neighboring cells (odd number of nodes, finer neighbors)
    receives the average of their traces.
    """
    
    col_0          = mesh.cols[col_key_0]
    cell_0         = col_0.cells[cell_key_0]
    [ndof_th_0]    = cell_0.ndofs[:]
    [th0_0, th1_0] = cell_0.pos[:]
    
    col_1          = mesh.cols[col_key_1]
    cell_1         = col_1.cells[cell_key_1]
    [ndof_th_1]    = cell_1.ndofs[:]
    [th0_1, th1_1] = cell_1.pos[:]
    
    # _0 <=> K in equations
    # _1 <=> K" in equations
    
    [_, _, _, _, thb_0, wth_0] = quad_xyth(nnodes_th = ndof_th_0)
    [_, _, _, _, thb_1, _    ] = quad_xyth(nnodes_th = ndof_th_1)
    E_th = np.zeros([ndof_th_1, ndof_th_0])
    
    # Nodes of K in physical space, and their Theta_F values
    thf_0 = push_forward(th0_0, th1_0, thb_0)
    Theta_F = Theta_F_func(thf_0, F)
    
    # Nodes of K pulled back to the reference interval of K". Clip so that a
    # node on the boundary of K" is not dropped by lag_eval due to round-off
    thb_0_1 = np.clip(pull_back(th0_1, th1_1, thf_0), -1., 1.)
    
    tol = 1.e-12 * (th1_0 - th0_0)
    for rr in range(0, ndof_th_0):
        th_r = thf_0[rr]
        if (th0_1 + tol < th_r) and (th_r < th1_1 - tol):
            # Node of K strictly inside K"
            share = 1.
        elif (np.abs(th_r - th0_1) <= tol) or (np.abs(th_r - th1_1) <= tol):
            # Node of K on the boundary between two neighboring cells K"
            share = 1. / 2.
        else:
            # Node of K not in K"
            continue
        
        wth_r = wth_0[rr]
        Theta_F_r = Theta_F[rr]
        for aa in range(0, ndof_th_1):
            xsi_ar = lag_eval(thb_1, aa, thb_0_1[rr])
            
            E_th[aa, rr] = share * wth_r * Theta_F_r * xsi_ar
            
    E_th[np.abs(E_th) < 1.e-15] = 0.0
    return E_th

# Theta^F function
def Theta_F_func(theta, F):
    return np.cos(theta - F * np.pi / 2.)
