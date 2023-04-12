import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import os, sys

sys.path.append('../../src')
import dg.quadrature as qd

from utils import print_msg

def test_8(quad_type = 'lg', dir_name = 'test_quad'):
    """
    Tests the projection of a Dirac-delta function onto a Legendre-Gauss,
    Legendre-Gauss-Lobatto basis.
    """

    dir_name = os.path.join(dir_name, 'test_8')
    os.makedirs(dir_name, exist_ok = True)

    if quad_type == 'lg':
        quad_type_str = 'Legendre-Gauss'

    elif quad_type == 'lgl':
        quad_type_str = 'Legendre-Gauss-Lobatto'
    else:
        print('ERROR: Test 8 recieved invalid quad_type. Please use "lg" or "lgl".')
        quit()

    # Set parameters, variables for test
    nnodes = 2**np.arange(4, 11, 1)
    ntrial = np.size(nnodes)
    xstar  = 0.25
    errors = np.zeros(ntrial)

    colors = ['#E69F00', '#56B4E9', '#009E73',
              '#F0E442', '#0072B2', '#D55E00',
              '#CC79A7']
    ncolor = len(colors)
    
    # Set up plot of Dirac-delta approximation so we plot each approximation
    fig, ax = plt.subplots()
    [ymin, ymax] = [10.**10, -10.**10]
        
    for trial in range(0, ntrial):
        perf_trial_0 = perf_counter()
        print_msg('[Trial {}] Starting...'.format(trial))
        
        nnode = nnodes[trial]
        
        if quad_type == 'lg':
            [nodes, weights] = qd.lg_quad(nnode)
        elif quad_type == 'lgl':
            [nodes, weights] = qd.lgl_quad(nnode)
        else:
            print('ERROR: Test 8 recieved invalid quad_type. Please use "lg" or "lgl".')
            quit()
            
        aprx_coeffs  = np.zeros([nnode])
        # Approximate the Dirac delta throughout the interval
        for ii in range(0, nnode):
            #dx = (10**(-3))
            #aprx_coeffs[ii] = dirac_aprx(xstar, 0.1, nodes[ii])
            #aprx_coeffs[ii] = step_func(xstar - dx, xstar + dx, nodes[ii])
            aprx_coeffs[ii] = np.amax([0.0, qd.lag_eval(nodes, ii, xstar)])
        
        # Plot the approximation
        ax.plot(nodes, aprx_coeffs, color = colors[trial%ncolor], linestyle = '-',
                label = '{}'.format(nnode))

        # Calculate the error of the integral
        intg = np.sum(weights * aprx_coeffs)
        errors[trial] = np.abs(intg - 1.)

        perf_trial_f    = perf_counter()
        perf_trial_diff = perf_trial_f - perf_trial_0
        msg = (
            '[Trial {}] Trial completed! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]\n'.format(perf_trial_diff)
        )
        print_msg(msg)

    # Save a plot of approximations
    ax.legend()
    title_str = ('Dirac-Delta Approximation on\n'
                 + '{} Nodal Bases\n').format(quad_type_str)
    ax.set_title(title_str)
    file_name = '{}_dirac_aprx.png'.format(quad_type)
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(dir_name, file_name), dpi = 300)
    plt.close(fig)
        
    # Plot errors
    fig, ax = plt.subplots()
    ax.plot(nnodes, errors,
            color = 'k', linestyle = '--')

    ax.set_yscale('log', base = 2)
    max_err = max(errors)
    min_err = min(errors)
    ymin = 2**(np.floor(np.log2(min_err)))
    ymax = 2**(np.ceil(np.log2(max_err)))
    ax.set_ylim([ymin, ymax])

    ax.set_xlabel('Number of Nodes')
    title_str = ('Dirac-Delta Approximation Integral Error on\n'
                 + '{} Nodal Bases\n').format(quad_type_str)
    ax.set_title(title_str)
    
    file_name = '{}_integration_acc.png'.format(quad_type)
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(dir_name, file_name), dpi = 300)
    plt.close(fig)

def dirac_aprx(xstar, a, x):
    return (1. / (np.abs(a) * np.sqrt(np.pi))) * np.exp(-((x - xstar) / a)**2)

def step_func(xmin, xmax, x):
    if (xmin <= x) and (x <= xmax):
        return 1
    else:
        return 0
