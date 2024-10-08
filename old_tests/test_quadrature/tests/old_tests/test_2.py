import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append("../../src")
import dg.quadrature as qd

def test_2(func, quad_type = "lg", dir_name = "test_quad"):
    """
    Plots the projection of an analytic function onto the Legendre-Gauss/
    Legendre-Gauss-Lobatto nodal basis of varying orders.
    """

    dir_name = os.path.join(dir_name, "test_2")
    os.makedirs(dir_name, exist_ok = True)
    
    if quad_type == "lg":
        quad_type_str = "Legendre-Gauss"

    elif quad_type == "lgl":
        quad_type_str = "Legendre-Gauss-Lobatto"

    else:
        print("ERROR: Test 2 recieved invalid quad_type. Please use "lg" or "lgl".")
        quit()

    min_power = 2
    max_power = 7
    
    # Plot the approximations as we go
    nx = 250
    xx = np.linspace(-1, 1, nx)
    fig, ax = plt.subplots()
    f_anl = func(xx)
    ax.plot(xx, f_anl, label = "Analytic",
            color = "k", linestyle = "-")
    
    
    powers = np.arange(min_power, max_power + 1, dtype = consts.INT)
    npowers = np.size(powers)
    nnodes_list = 2**powers

    # Construct projection and plot approximations
    colors = ["#E69F00", "#56B4E9", "#009E73",
              "#F0E442", "#0072B2", "#D55E00",
              "#CC79A7"]
    for nn in range(0, npowers):
        nnodes = nnodes_list[nn]

        # Calculate analytic reconstruction of low-order projection
        if quad_type == "lg":
            [nodes, _] = qd.lg_quad(nnodes)
        elif quad_type == "lgl":
            [nodes, _] = qd.lgl_quad(nnodes)
        else:
            print("ERROR: Test 2 recieved invalid quad_type. Please use "lg" or "lgl".")
            quit()
        f_proj = func(nodes)
        f_proj_anl = np.zeros([nx])
        for x_idx in range(0, nx):
            for ii in range(0, nnodes):
                f_proj_anl[x_idx] += f_proj[ii] \
                    * qd.lag_eval(nodes, ii, xx[x_idx])

        # Plot analytic reconstruction
        lbl = "{} Nodes".format(nnodes)
        ax.plot(xx, f_proj_anl, label = lbl,
                color = colors[nn], linestyle = "-")

    
    ax.legend()
    title_str = ("1-D Function Projection Comparison\n"
                 + "{} Nodal Basis").format(quad_type_str)
    ax.set_title(title_str)

    file_name = "{}_proj_comp.png".format(quad_type)
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(dir_name, file_name), dpi = 300)
    plt.close(fig)
