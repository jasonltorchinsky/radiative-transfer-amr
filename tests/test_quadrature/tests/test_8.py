import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append('../../src')
import dg.quadrature as qd

def test_8(func, func_ddx, dir_name = 'test_quad'):
    """
    Plots the projection of the derviative an analytic function onto
    the Legendre-Gauss nodal basis of varying orders.
    """

    min_power = 2
    max_power = 7
    
    # Plot the approximations as we go
    nx = 250
    xx = np.linspace(-1, 1, nx)
    fig, ax = plt.subplots()
    f_ddx_anl = func_ddx(xx)
    ax.plot(xx, f_ddx_anl, label = 'Analytic',
            color = 'k', linestyle = '-')
    
    
    powers = np.arange(min_power, max_power + 1, dtype = np.int32)
    npowers = np.size(powers)
    nnodes_list = 2**powers

    # Construct projection and plot approximations
    colors = ['#E69F00', '#56B4E9', '#009E73',
              '#F0E442', '#0072B2', '#D55E00',
              '#CC79A7']
    for nn in range(0, npowers):
        nnodes = nnodes_list[nn]

        # Calculate analytic reconstruction of low-order projection
        [nodes, _] = qd.lg_quad(nnodes)
        ddx = qd.lag_ddx(nodes)
        f_proj = func(nodes)
        f_ddx_proj = ddx @ f_proj
        f_ddx_proj_anl = np.zeros([nx])
        for x_idx in range(0, nx):
            for ii in range(0, nnodes):
                f_ddx_proj_anl[x_idx] += f_ddx_proj[ii] \
                    * qd.lag_eval(nodes, ii, xx[x_idx])

        # Plot analytic reconstruction
        lbl = '{} Nodes'.format(nnodes)
        ax.plot(xx, f_ddx_proj_anl, label = lbl,
                color = colors[nn], linestyle = '-')

    
    ax.legend()
    title_str = ('1-D Function Derivative Projection Comparison\n'
                 + 'Legendre-Gauss Nodal Basis')
    ax.set_title(title_str)

    file_name = 'lg_ddx_proj_comp.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(dir_name, file_name), dpi = 300)
    plt.close(fig)
