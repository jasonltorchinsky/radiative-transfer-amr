import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from time import perf_counter
import os, sys

from .gen_mesh import gen_mesh

sys.path.append("../../src")
from dg.mesh import get_cell_nhbr_in_col
import dg.quadrature as qd
from dg.mesh.utils import plot_mesh_p
from dg.projection import push_forward, pull_back, get_f2f_matrix
from rt import get_Eth
from amr import rand_err, ref_by_ind

from utils import print_msg


def test_5(dir_name = "test_rt"):
    """
    Eth matrix generation.
    TO-DO: MAKE CHECK FOR DIFFERENT P.
    """
    
    test_dir = os.path.join(dir_name, "test_5")
    os.makedirs(test_dir, exist_ok = True)

    do_plot_mesh = True
    
    # Construct the F2F matrix manually.
    # _0 correspond to cell we get basis functions from
    # _1 corresponds to cell we get nodes from
    rng = np.random.default_rng()
    
    [Lx, Ly]                   = [2., 3.]
    pbcs                       = [False, False]
    [ndof_x, ndof_y, ndof_th]  = list(rng.integers(low = 3, high = 10, size = 3))
    has_th                     = True
    mesh = gen_mesh(Ls     = [Lx, Ly],
                    pbcs   = pbcs,
                    ndofs  = [3, 3, 3],
                    has_th = has_th)

    nref = 3
    for ref in range(0, nref):
        rand_err_ind = rand_err(mesh, kind = "all", form = "h")
        mesh = ref_by_ind(mesh, rand_err_ind,
                          ref_ratio = 0.5,
                          form = "h")

    if do_plot_mesh:
        file_name = os.path.join(test_dir, "mesh_3d.png")
        plot_mesh_p(mesh      = mesh,
                    file_name = file_name,
                    plot_dim  = 3)
        file_name = os.path.join(test_dir, "mesh_2d.png")
        plot_mesh_p(mesh        = mesh,
                    file_name   = file_name,
                    plot_dim    = 2,
                    label_cells = (nref <= 3))

    do_fail = False
    col_items = sorted(mesh.cols.items())
    col_lfs = ((col_key, col) for col_key, col in col_items if col.is_lf)
    for col_key_0, col_0 in col_lfs:
        cell_items_0 = sorted(col_0.cells.items())
        cell_lfs_0 = ((cell_key, cell) for cell_key, cell in cell_items_0 if cell.is_lf)
        for F in range(0, 4):
            col_nhbr_keys = (col_key_1 for col_key_1 in list(set(col_0.nhbr_keys[F]))
                             if col_key_1 is not None)
            
            for col_key_1 in col_nhbr_keys:
                col_1 = mesh.cols[col_key_1]
                if col_1.is_lf:
                    for cell_key_0, cell_0 in cell_lfs_0:
                        [th0_0, th1_0] = cell_0.pos[:]
                        [ndof_th_0] = cell_0.ndofs[:]
                        [_, _, _, _, thb_0, wth_0] = \
                            qd.quad_xyth(nnodes_th = ndof_th_0)
                        
                        thf_0 = push_forward(th0_0, th1_0, thb_0)
                        Theta_F = Theta_F_func(thf_0, F)
                        
                        cell_nhbr_keys = \
                            get_cell_nhbr_in_col(mesh = mesh,
                                                 col_key = col_key_0,
                                                 cell_key = cell_key_0,
                                                 nhbr_col_key = col_key_1)
                        cell_nhbr_keys = (cell_nhbr_key for cell_nhbr_key in cell_nhbr_keys
                                          if cell_nhbr_key is not None)
                        
                        for cell_key_1 in cell_nhbr_keys:
                            cell_1 = col_1.cells[cell_key_1]
                            if cell_1.is_lf:
                                [th0_1, th1_1] = cell_1.pos[:]
                                [ndof_th_1] = cell_1.ndofs[:]
                                [_, _, _, _, thb_1, wth_1] = \
                                    qd.quad_xyth(nnodes_th = ndof_th_1)
                                
                                thb_0_1 = pull_back(th0_1, th1_1, thf_0)
                                Eth_mat = np.zeros([ndof_th_1, ndof_th_0])
                                for aa in range(0, ndof_th_1):
                                    for rr in range(0, ndof_th_0):
                                        f2f_ar = qd.lag_eval(thb_1, aa, thb_0_1[rr])
                                        if np.abs(f2f_ar) > 1.e-14:
                                            Eth_mat[aa, rr] = wth_0[rr] * Theta_F[rr] * f2f_ar
                                            
                                Eth = get_Eth(mesh,
                                              col_key_0,
                                              cell_key_0,
                                              col_key_1,
                                              cell_key_1,
                                              F)
                                            
                                err = np.amax(np.abs(Eth_mat - Eth))
                                if err > 1.e-13:
                                    msg = ("FAILED: [{}, {}], ".format(col_key_0, cell_key_0) +
                                           "[{}, {}]".format(col_key_1, cell_key_1)
                                           )
                                    print(msg)
                                    print(Eth_mat)
                                    print(Eth)
                                    print("\n")
                                    E = get_Eth(mesh,
                                                col_key_0,
                                                cell_key_0,
                                                col_key_1,
                                                cell_key_1,
                                                F)
                                    
                                    do_fail = True
    
    if do_fail:
        print_msg("Test failed!\n")
    else:
        print_msg("Test passed!\n")

# Theta^F function
def Theta_F_func(theta, F):
    return np.cos(theta - F * np.pi / 2)
