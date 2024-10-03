import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from time import perf_counter
import os, sys

from .gen_mesh import gen_mesh

sys.path.append("../../src")
import dg.quadrature as qd
from dg.mesh.utils import plot_mesh_p
from dg.projection import push_forward, pull_back, get_f2f_matrix
from rt import get_Ex, get_Ey
from amr import rand_err, ref_by_ind

from utils import print_msg


def test_4(dir_name = "test_rt"):
    """
    Ex matrix generation
    """
    
    test_dir = os.path.join(dir_name, "test_4")
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
        rand_err_ind = rand_err(mesh, kind = "spt", form = "hp")
        mesh = ref_by_ind(mesh, rand_err_ind,
                          ref_ratio = 0.5,
                          form = "hp")

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
    dims = ["x", "y"]
    for dim in dims:
        for col_key_0, col_0 in col_items:
            if col_0.is_lf:
                if dim == "x":
                    [n0_0, _, n1_0, _] = col_0.pos[:]
                    [ndof_0, _]        = col_0.ndofs[:]
                    [nnb_0, wn_0, _, _, _, _] = qd.quad_xyth(nnodes_x = ndof_0)

                    col_nhbr_keys = col_0.nhbr_keys[1] + col_0.nhbr_keys[3]
                elif dim == "y":
                    [_, n0_0, _, n1_0] = col_0.pos[:]
                    [_, ndof_0]        = col_0.ndofs[:]
                    [_, _, nnb_0, wn_0, _, _] = qd.quad_xyth(nnodes_y = ndof_0)

                    col_nhbr_keys = col_0.nhbr_keys[0] + col_0.nhbr_keys[2]

                nnf_0 = push_forward(n0_0, n1_0, nnb_0)
                wn_0  = wn_0.reshape([1, ndof_0])

                
                col_nhbr_keys = list(set(col_nhbr_keys))
                for col_key_1 in col_nhbr_keys:
                    if col_key_1 is not None:
                        col_1 = mesh.cols[col_key_1]
                        if col_1.is_lf:
                            if dim == "x":
                                [n0_1, _, n1_1, _] = col_1.pos[:]
                                [ndof_1, _]        = col_1.ndofs[:]
                                [nnb_1, wn_1, _, _, _, _] = \
                                    qd.quad_xyth(nnodes_x = ndof_1)
                            elif dim == "y":
                                [_, n0_1, _, n1_1] = col_1.pos[:]
                                [_, ndof_1]        = col_1.ndofs[:]
                                [_, _, nnb_1, wn_1, _, _] = \
                                    qd.quad_xyth(nnodes_y = ndof_1)
                            
                            nnb_0_1 = pull_back(n0_1, n1_1, nnf_0)

                            if ndof_0 >= ndof_1:
                                f2f_mat = np.zeros([ndof_1, ndof_0])
                                for cc in range(0, ndof_1):
                                    for dd in range(0, ndof_0):
                                        f2f_cd = qd.lag_eval(nnb_1, cc, nnb_0_1[dd])
                                        if np.abs(f2f_cd) > 1.e-14:
                                            f2f_mat[cc, dd] = f2f_cd
                                            
                                E_mat = wn_0 * f2f_mat
                            else:
                                f2f_mat_0 = np.zeros([ndof_1, ndof_1])
                                nnf_1_0 = push_forward(n0_0, n1_0, nnb_1)
                                nnb_0_1_0 = pull_back(n0_1, n1_1, nnf_1_0)
                                for cc in range(0, ndof_1):
                                    for ccp in range(0, ndof_1):
                                        f2f_ccp = \
                                            qd.lag_eval(nnb_1, cc, nnb_0_1_0[ccp])
                                        if np.abs(f2f_ccp) > 1.e-14:
                                            f2f_mat_0[cc, ccp] = f2f_ccp

                                f2f_mat_1 = np.zeros([ndof_0, ndof_1])
                                for dd in range(0, ndof_0):
                                    for ccp in range(0, ndof_1):
                                        f2f_dcp = \
                                            qd.lag_eval(nnb_0, dd, nnb_1[ccp])
                                        if np.abs(f2f_dcp) > 1.e-14:
                                            f2f_mat_1[dd, ccp] = f2f_dcp

                                E_mat = np.zeros([ndof_1, ndof_0])
                                for cc in range(0, ndof_1):
                                    for dd in range(0, ndof_0):
                                        for ccp in range(0, ndof_1):
                                            E_mat[cc, dd] += wn_1[ccp] \
                                                * f2f_mat_0[cc, ccp] \
                                                * f2f_mat_1[dd, ccp]
                            
                            if dim == "x":
                                E = get_Ex(mesh, col_key_0, col_key_1)
                            elif dim == "y":
                                E = get_Ey(mesh, col_key_0, col_key_1)
                        
                            err = np.amax(np.abs(E_mat - E))
                            if err > 1.e-13:
                                print("FAILED: {}, {}, {}".format(dim, col_key_0, col_key_1))
                                print(E_mat)
                                print(E)

                                if dim == "x":
                                    E = get_Ex(mesh, col_key_0, col_key_1)
                                elif dim == "y":
                                    E = get_Ey(mesh, col_key_0, col_key_1)
                                
                                do_fail = True
    
    if do_fail:
        print_msg("Test failed!\n")
    else:
        print_msg("Test passed!\n")
