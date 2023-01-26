import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append('../../src')
import dg.quadrature as qd

def test_7(func, dir_name = 'test_quad'):
    """
    Tests the order of convergence for projecting an analytic function
    onto a Legengre-Gauss-Lobatto basis.
    """

    nnodes_list = np.arange(2**2, 2**7 + 1, 8)
    ntrials = np.size(nnodes_list)
    
    # Calculate error using a high-order quadrature rule
    max_nnodes = nnodes_list[-1]
    quad_nnodes = 2 * max_nnodes
    [quad_nodes, quad_weights] = qd.lgl_quad(quad_nnodes)
    f_anl = func(quad_nodes)
    
    error_L1 = np.zeros([ntrials])
    error_L2 = np.zeros([ntrials])

    # Construct projection, calculate error
    for nn in range(0, ntrials):
        nnodes = nnodes_list[nn]

        # Calculate analytic reconstruction of low-order projection
        [nodes, _] = qd.lgl_quad(nnodes)
        f_proj = func(nodes)
        f_proj_anl = np.zeros_like(quad_nodes)
        for x_idx in range(0, quad_nnodes):
            for ii in range(0, nnodes):
                f_proj_anl[x_idx] += f_proj[ii] \
                    * qd.lag_eval(nodes, ii, quad_nodes[x_idx])

        # Calculate error
        tally = 0
        for x_idx in range(0, quad_nnodes):
            error_L1[nn] += quad_weights[x_idx] \
                * np.abs(f_anl[x_idx] - f_proj_anl[x_idx])
            error_L2[nn] += quad_weights[x_idx] \
                * (f_anl[x_idx] - f_proj_anl[x_idx])**2

        error_L2[nn] = np.sqrt(error_L2[nn])
        
    # Plot all errors
    fig, ax = plt.subplots()
    ax.plot(nnodes_list, error_L1,  label = '$L^1$ Error',
                color = 'k', linestyle = '-')
    ax.plot(nnodes_list, error_L2,  label = '$L^2$ Error',
              color = 'b', linestyle = '-')

    ax.set_xscale('log', base = 2)
    ax.set_yscale('log', base = 10)
    
    ax.legend()
    title_str = ('1-D Function Projection Accuracy\n'
                 + 'Legendre-Gauss-Lobatto Nodal Basis')
    ax.set_title(title_str)
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Error')

    file_name = 'lgl_proj_acc.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(dir_name, file_name), dpi = 300)
    plt.close(fig)
