import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve, eigs
import os, sys

sys.path.append('../../src')
import dg.quadrature as qd
from dg.projection import push_forward, pull_back

def test_0(dir_name = 'test_rt'):
    """
    Test the push-forward, pull-back maps.
    """
    
    test_dir = os.path.join(dir_name, 'test_0')
    os.makedirs(test_dir, exist_ok = True)

    # Create simplified scenario of three elements sharing a boundary
    ndof_x_0 = 5
    ndof_x_1 = 4
    ndof_x_2 = 3

    [x0_0, x1_0] = [0., 2.]
    [x0_1, x1_1] = [0., 1.]
    [x0_2, x1_2] = [1., 2.]

    [nodes_x_0, _, _, _ ,_, _] = qd.quad_xyth(nnodes_x = ndof_x_0)
    [nodes_x_1, _, _, _ ,_, _] = qd.quad_xyth(nnodes_x = ndof_x_1)
    [nodes_x_2, _, _, _ ,_, _] = qd.quad_xyth(nnodes_x = ndof_x_2)

    nx  = 500
    xxb = np.linspace(-1, 1, nx)

    ## Level l+1 cell basis to level l cell domain (left = 1)
    xxf_0   = push_forward(x0_0, x1_0, xxb)
    xxb_0_1 = pull_back(   x0_1, x1_1, xxf_0)

    phi_pi = np.zeros([ndof_x_1, nx])
    for pp in range(0, ndof_x_1):
        for ii in range(0, nx):
            phi_pi[pp][ii] = qd.lag_eval(nodes_x_1, pp, xxb_0_1[ii])
    
    # Create plots
    fig, ax = plt.subplots()
    colors = ['#E69F00', '#56B4E9', '#009E73',
              '#F0E442', '#0072B2', '#D55E00',
              '#CC79A7']
    for pp in range(0, ndof_x_1):
        ax.plot(xxb, phi_pi[pp],
                color = colors[pp],
                linestyle = '-',
                label = 'Basis Function {}'.format(pp))

    ax.legend()
    
    ax.set_title('Level $(l + 1)$ Cell Basis to Level $l$ Cell Domain')
    
    file_name = 'l+1_to_l_1.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(test_dir, file_name), dpi = 300)
    plt.close(fig)

    ## Level l+1 cell basis to level l cell domain (right = 2)
    xxf_0   = push_forward(x0_0, x1_0, xxb)
    xxb_0_2 = pull_back(   x0_2, x1_2, xxf_0)

    phi_pi = np.zeros([ndof_x_2, nx])
    for pp in range(0, ndof_x_2):
        for ii in range(0, nx):
            phi_pi[pp][ii] = qd.lag_eval(nodes_x_2, pp, xxb_0_2[ii])
    
    # Create plots
    fig, ax = plt.subplots()
    colors = ['#E69F00', '#56B4E9', '#009E73',
              '#F0E442', '#0072B2', '#D55E00',
              '#CC79A7']
    for pp in range(0, ndof_x_2):
        ax.plot(xxb, phi_pi[pp],
                color = colors[pp],
                linestyle = '-',
                label = 'Basis Function {}'.format(pp))

    ax.legend()
    
    ax.set_title('Level $(l + 1)$ Cell Basis to Level $l$ Cell Domain')
    
    file_name = 'l+1_to_l_2.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(test_dir, file_name), dpi = 300)
    plt.close(fig)


    ## Level l cell basis to level l+1 cell domain
    xxf_1   = push_forward(x0_1, x1_1, xxb)
    xxb_1_0 = pull_back(   x0_0, x1_0, xxf_1)

    phi_pi = np.zeros([ndof_x_0, nx])
    for pp in range(0, ndof_x_0):
        for ii in range(0, nx):
            phi_pi[pp][ii] = qd.lag_eval(nodes_x_0, pp, xxb_1_0[ii])
    
    # Create plots
    fig, ax = plt.subplots()
    colors = ['#E69F00', '#56B4E9', '#009E73',
              '#F0E442', '#0072B2', '#D55E00',
              '#CC79A7']
    for pp in range(0, ndof_x_0):
        ax.plot(xxb, phi_pi[pp],
                color = colors[pp],
                linestyle = '-',
                label = 'Basis Function {}'.format(pp))

    ax.legend()
    
    ax.set_title('Level $l$ Cell Basis to Level $(l + 1)$ Cell Domain')
    
    file_name = 'l_to_l+1.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(test_dir, file_name), dpi = 300)
    plt.close(fig)
