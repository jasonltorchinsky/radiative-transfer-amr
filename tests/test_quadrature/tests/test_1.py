import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append('../../tests')
from test_cases import get_cons_funcs

sys.path.append('../../src')
import dg.quadrature as qd

def test_1(quad_type = 'lg', dir_name = 'test_quad', **kwargs):
    """
    Tests the projection of an analytic function f onto the Legendre-Gauss/
    Legendre-Gauss-Lobatto basis.
    """

    default_kwargs = {}
    kwargs = {**default_kwargs, **kwargs}
    
    test_dir = os.path.join(dir_name, 'test_1')
    os.makedirs(test_dir, exist_ok = True)

    [_, f, _] = get_cons_funcs(func_num = 0)

    if quad_type == 'lg':
        quad_type_str = 'Legendre-Gauss'
    elif quad_type == 'lgr':
        quad_type_str = 'Legendre-Gauss-Radau'
    elif quad_type == 'lgl':
        quad_type_str = 'Legendre-Gauss-Lobatto'
    elif quad_type == 'uni':
        quad_type_str = 'Uniform'
        
    # Set up grid for plotting projections
    nx = 250
    xx = np.linspace(-1, 1, nx)
    f_anl  = f(xx)

    f_projs = {}
    nnodes = [4, 8, 16, 32]
    for nnode in nnodes:
        if quad_type == 'lg':
            [nodes, weights] = qd.lg_quad(nnode)
        elif quad_type == 'lgr':
            [nodes, weights] = qd.lgr_quad(nnode)
            print(nnode)
            print(nodes)
            print(weights)
            print('\n')
        elif quad_type == 'lgl':
            [nodes, weights] = qd.lgl_quad(nnode)
        elif quad_type == 'uni':
            [nodes, weights] = qd.uni_quad(nnode)
            
        else:
            msg = (
                'ERROR: Test 1 recieved invalid quad_type. ' +
                'Please use "lg", "lgr", "lgl", "uni".'
            )
            print(msg)
            quit()
    
        # Calculate f on nodes for projection
        f_nodes = f(nodes)
        
        # Calculate f throughout the interval
        f_proj = np.zeros_like(xx)
        
        for x_idx in range(0, nx):
            for ii in range(0, nnode):
                f_proj[x_idx] += f_nodes[ii] * qd.lag_eval(nodes, ii, xx[x_idx])

        f_projs[nnode] = f_proj
                
    # Plot all functions for comparison
    colors = ['#E69F00', '#56B4E9', '#009E73',
              '#F0E442', '#0072B2', '#D55E00',
              '#CC79A7', '#882255']
    ncolors = len(colors)
    
    fig, ax = plt.subplots()

    f_proj_items = sorted(f_projs.items())
    c_idx = 0
    for nnode, f_proj in f_proj_items:
        ax.plot(xx, f_proj,
                label = 'Order {}'.format(nnode),
                color = colors[c_idx%ncolors], linestyle = '-')
        c_idx += 1
    
    ax.plot(xx, f_anl,  label = 'Analytic',
            color = 'k', linestyle = '-')

    ax.set_xlim([-1.1, 1.1])
    df = np.amax(f_anl) - np.amin(f_anl)
    ax.set_ylim([np.amin(f_anl) - 0.1 * df, np.amax(f_anl) + 0.1 * df]) 
    
    ax.legend()
    title_str = (
        '1-D Function Projection To {} Nodal Basis'.format(quad_type_str)
        )
    ax.set_title(title_str)
    
    file_name = '{}_proj_1d.png'.format(quad_type)
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(test_dir, file_name), dpi = 300)
    plt.close(fig)


def zero(x):
    return 0
