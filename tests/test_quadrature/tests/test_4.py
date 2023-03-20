import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append('../../src')
import dg.quadrature as qd

def test_4(func, func_ddx, quad_type = 'lg', dir_name = 'test_quad'):
    """
    Plots the projection of the derviative an analytic function onto
    the Legendre-Gauss/Legendre-Gauss-Lobatto  nodal basis of varying orders.
    """

    dir_name = os.path.join(dir_name, 'test_4')
    os.makedirs(dir_name, exist_ok = True)
    
    if quad_type == 'lg':
        quad_type_str = 'Legendre-Gauss'

    elif quad_type == 'lgl':
        quad_type_str = 'Legendre-Gauss-Lobatto'

    else:
        print('ERROR: Test 4 recieved invalid quad_type. Please use "lg" or "lgl".')
        quit()
    
    min_power = 2
    max_power = 6
    
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
        if quad_type == 'lg':
            [nodes, _] = qd.lg_quad(nnodes)
        elif quad_type == 'lgl':
            [nodes, _] = qd.lgl_quad(nnodes)
        else:
            print('ERROR: Test 4 recieved invalid quad_type. Please use "lg" or "lgl".')
            quit()

        f_proj = func(nodes)
        #ddx = qd.lag_ddx(nodes)
        #f_ddx_proj = ddx @ f_proj
        f_ddx_proj_anl = np.zeros([nx])
        for x_idx in range(0, nx):
            for ii in range(0, nnodes):
                #f_ddx_proj_anl[x_idx] += f_ddx_proj[ii] \
                #    * qd.lag_eval(nodes, ii, xx[x_idx])
                f_ddx_proj_anl[x_idx] += f_proj[ii] \
                    * qd.lag_ddx_eval(nodes, ii, xx[x_idx])

        # Plot analytic reconstruction
        lbl = '{} Nodes'.format(nnodes)
        ax.plot(xx, f_ddx_proj_anl, label = lbl,
                color = colors[nn], linestyle = '-')

    
    ax.legend()
    title_str = ('1-D Function Derivative Projection Comparison\n'
                 + '{} Nodal Basis').format(quad_type_str)
    ax.set_title(title_str)

    file_name = '{}_ddx_proj_comp.png'.format(quad_type)
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(dir_name, file_name), dpi = 300)
    plt.close(fig)
