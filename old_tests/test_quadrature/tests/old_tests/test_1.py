import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append("../../src")
import dg.quadrature as qd

def test_1(func, src_nnodes = 41, trgt_nnodes = 10,
           quad_type = "lg", dir_name = "test_quad"):
    """
    Tests the projection of an analytic function f onto the Legendre-Gauss/
    Legendre-Gauss-Lobatto basis.
    """

    dir_name = os.path.join(dir_name, "test_1")
    os.makedirs(dir_name, exist_ok = True)

    if quad_type == "lg":
        [src_nodes, src_weights] = qd.lg_quad(src_nnodes)
        [trgt_nodes, trgt_weights] = qd.lg_quad(trgt_nnodes)
        quad_type_str = "Legendre-Gauss"

    elif quad_type == "lgl":
        [src_nodes, src_weights] = qd.lgl_quad(src_nnodes)
        [trgt_nodes, trgt_weights] = qd.lgl_quad(trgt_nnodes)
        quad_type_str = "Legendre-Gauss-Lobatto"

    else:
        print("ERROR: Test 1 recieved invalid quad_type. Please use "lg" or "lgl".")
        quit()
        

    # Calculate f on source nodes, project onto target nodes
    f_src = func(src_nodes)
    proj_mtx = qd.calc_proj_mtx_1d(src_nodes, trgt_nodes)
    f_proj = proj_mtx @ f_src

    # Calculate f on target nodes for comparison
    f_trgt = func(trgt_nodes)

    # Calculate f throughout the interval for each
    nx = 250
    xx = np.linspace(-1, 1, nx)
    f_anl = func(xx)
    f_src_anl  = np.zeros_like(xx)
    f_proj_anl = np.zeros_like(xx)
    f_trgt_anl = np.zeros_like(xx)
    
    for x_idx in range(0, nx):
        for ii in range(0, src_nnodes):
            f_src_anl[x_idx] += f_src[ii] * qd.lag_eval(src_nodes, ii, xx[x_idx])

    for x_idx in range(0, nx):
        for jj in range(0, trgt_nnodes):
            f_proj_anl[x_idx] += f_proj[jj] * qd.lag_eval(trgt_nodes, jj, xx[x_idx])
            f_trgt_anl[x_idx] += f_trgt[jj] * qd.lag_eval(trgt_nodes, jj, xx[x_idx])

    # Plot all functions for comparison
    fig, ax = plt.subplots()
    src_err  = np.abs(f_anl - f_src_anl)
    proj_err = np.abs(f_anl - f_proj_anl)
    trgt_err = np.abs(f_anl - f_trgt_anl)
    ax.plot(xx, src_err,  label = "|Analytic - Source|",
            color = "k", linestyle = "-")
    ax.plot(xx, proj_err, label = "|Analytic - Projection|",
            color = "b", linestyle = "-")
    ax.plot(xx, trgt_err, label = "|Analytic - Target|",
            color = "r", linestyle = "-")
    ax.legend()
    title_str = ("1-D Function Projection Between\n"
                 + "{} Nodal Bases\n"
                 + "Source Order: {}\n"
                 + "Target Order: {}").format(quad_type_str, src_nnodes,
                                              trgt_nnodes)
    ax.set_title(title_str)

    file_name = "{}_proj_1d_{:03d}_{:03d}.png".format(quad_type,
                                                      src_nnodes, trgt_nnodes)
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(dir_name, file_name), dpi = 300)
    plt.close(fig)
