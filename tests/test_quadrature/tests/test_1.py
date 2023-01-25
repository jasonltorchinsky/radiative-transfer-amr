import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append('../../src')
import dg.quadrature as qd

def test_1(order = 5, dir_name = 'test_quad'):
    """
    Creates plots of the spatial basis functions.
    """

    # The number of nodes needed is equal to the order of approximation
    # desired for Legendre-Gauss-Lobatto quadrature
    nnodes = order
    [nodes, weights] = qd.lgl_quad(nnodes)

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
    title_str = ('Legendre-Gauss-Lobatto Nodes\n' +
                 'Lagrange Polynomial Basis Functions\n' +
                 'Order {}').format(order)
    ax.set_title(title_str)

    file_name = 'lgl_basis.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(dir_name, file_name), dpi = 300)
    plt.close(fig)
