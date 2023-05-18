import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append('../../src')
import dg.quadrature as qd

def test_0(nnodes = 5, quad_type = 'lg', dir_name = 'test_quad'):
    """
    Creates plots of the spatial basis functions (Legendre polynomials
    interpolating the Legendre-Gauss/Legendre-Gauss-Lobatto nodes).
    """

    dir_name = os.path.join(dir_name, 'test_0')
    os.makedirs(dir_name, exist_ok = True)

    if quad_type == 'lg':
        [nodes, weights] = qd.lg_quad(nnodes)
        quad_type_str = 'Legendre-Gauss'
    elif quad_type == 'lgr':
        [nodes, weights] = qd.lgr_quad(nnodes)
        quad_type_str = 'Legendre-Gauss-Radau'
    elif quad_type == 'lgl':
        [nodes, weights] = qd.lgl_quad(nnodes)
        quad_type_str = 'Legendre-Gauss-Lobatto'
    elif quad_type == 'uni':
        [nodes, weights] = qd.uni_quad(nnodes)
        quad_type_str = 'Uniform'
    else:
        msg = (
            'ERROR: Test 0 recieved invalid quad_type. ' +
            'Please use "lg", "lgr", "lgl", "uni".'
        )
        print(msg)
        quit()

    # Plot just the grid points
    fig, ax = plt.subplots()
    ax.axvline(x = -1, color = 'gray')
    ax.axvline(x = 1, color = 'gray')
    yy = np.ones_like(nodes)
    ax.scatter(nodes, yy,
               color = 'k')
    ax.set_xlim([-1.1, 1.1])
    ax.set_yticks([])
    title_str = '{}-Order {} Nodes\n'.format(nnodes, quad_type_str)
    ax.set_title(title_str)

    file_name = '{}_nodes.png'.format(quad_type)
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(dir_name, file_name), dpi = 300)
    plt.close(fig)

    nx = 500
    xx = np.linspace(-1.1, 1.1, nx)
    basis_funcs = np.zeros([nnodes, nx])
    for ii in range(0, nnodes):
        for x_idx in range(0, nx):
            basis_funcs[ii, x_idx] = qd.lag_eval(nodes, ii, xx[x_idx])

    fig, ax = plt.subplots()
    for ii in range(0, nnodes):
        lbl = 'Basis Function {}'.format(ii)
        ax.plot(xx, basis_funcs[ii, :], label = lbl,
            linestyle = '-')
    ax.legend()
    title_str = ('{} Nodes\n' +
                 'Lagrange Polynomial Basis Functions\n' +
                 'Order {}').format(quad_type_str, nnodes)
    ax.set_title(title_str)

    file_name = '{}_basis.png'.format(quad_type)
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(dir_name, file_name), dpi = 300)
    plt.close(fig)
