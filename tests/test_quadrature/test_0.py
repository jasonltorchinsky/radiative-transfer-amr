import numpy as np
import matplotlib.pyplot as plt

def test_0(f, g, order = 5, dir_name = dir_name):
    """
    Tests Legendre-Gauss node and quadrature weight calculation.

    Tests Gauss-Lobatto basis function evaluation (i.e., the Lagrange polynomial
    interpolating the Gauss-Lobatto 
    """

    # Test LGL nodes, evaluating GL
    #[nodes, weights, _] = qd.lgl_quad(order)
    [nodes, weights] = qd.lag_quad(order + 1, [-1, 1])

    nx = 500
    xx = np.linspace(-1, 1, nx)
    gl_funcs = np.zeros([order + 1, nx])
    for i in range(0, order + 1):
        for x_idx in range(0, nx):
            gl_funcs[i, x_idx] = qd.gl_eval(nodes, i, xx[x_idx])

    fig, ax = plt.subplots()
    for i in range(0, order + 1):
        lbl = 'Basis Function {}'.format(i)
        ax.plot(xx, gl_funcs[i, :], label = lbl,
            linestyle = '-')
    ax.legend()

    file_name = 'lg_basis.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(dir_name, file_name), dpi = 300)
    plt.close(fig)
