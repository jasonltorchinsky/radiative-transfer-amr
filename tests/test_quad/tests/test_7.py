import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append('../../src')
import dg.quadrature as qd

def test_7(func, Func,
           quad_type = 'lg',
           dir_name = 'test_quad'):
    """
    Tests the integration of an analytic function f onto the Legendre-Gauss/
    Legendre-Gauss-Lobatto basis.
    """

    dir_name = os.path.join(dir_name, 'test_7')
    os.makedirs(dir_name, exist_ok = True)

    nnodes = 2**np.arange(2, 12, 1)

    if quad_type == 'lg':
        quad_type_str = 'Legendre-Gauss'

    elif quad_type == 'lgl':
        quad_type_str = 'Legendre-Gauss-Lobatto'
    else:
        print('ERROR: Test 7 recieved invalid quad_type. Please use "lg" or "lgl".')
        quit()

    ntrial = np.size(nnodes)
    anl_intgs = np.full(ntrial, Func(1) - Func(-1))
    apr_intgs = np.zeros(ntrial)
    errors    = np.zeros(ntrial)
        
    for trial in range(0, ntrial):
        nnode = nnodes[trial]
        
        if quad_type == 'lg':
            [nodes, weights] = qd.lg_quad(nnode)
        elif quad_type == 'lgl':
            [nodes, weights] = qd.lgl_quad(nnode)
        else:
            print('ERROR: Test 6 recieved invalid quad_type. Please use "lg" or "lgl".')
            quit()        
    
        # Calculate F throughout the interval
        for ii in range(0, nnode):
            apr_intgs[trial] += weights[ii] * func(nodes[ii])

        errors[trial] = np.abs(anl_intgs[trial] - apr_intgs[trial])
        
    # Plot errors
    fig, ax = plt.subplots()
    ax.plot(nnodes, errors,
            color = 'k', linestyle = '--')

    ax.set_xscale('log', base = 2)
    ax.set_yscale('log', base = 10)

    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('|Analytic - Approximate|')

    title_str = ('1-D Function Integration on\n'
                 + '{} Nodal Bases\n').format(quad_type_str)
    ax.set_title(title_str)
    
    file_name = '{}_integration_acc.png'.format(quad_type)
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(dir_name, file_name), dpi = 300)
    plt.close(fig)
