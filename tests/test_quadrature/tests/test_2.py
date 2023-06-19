import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append('../../tests')
from test_cases import get_cons_funcs

sys.path.append('../../src')
import dg.quadrature as qd

def test_2(quad_type = 'lg', dir_name = 'test_quad', **kwargs):
    """
    Tests convergence of the quadrature rule.
    """

    default_kwargs = {}
    kwargs = {**default_kwargs, **kwargs}
    
    test_dir = os.path.join(dir_name, 'test_2')
    os.makedirs(test_dir, exist_ok = True)

    if quad_type == 'lg':
        quad_type_str = 'Legendre-Gauss'
    elif quad_type == 'lgr':
        quad_type_str = 'Legendre-Gauss-Radau'
    elif quad_type == 'lgl':
        quad_type_str = 'Legendre-Gauss-Lobatto'
    elif quad_type == 'uni':
        quad_type_str = 'Uniform'

    max_ntrial = 12
    for func_num in range(0, 6):
        [F, f, _] = get_cons_funcs(func_num = func_num)
        
        nnodes = [0] * max_ntrial
        errs   = [0] * max_ntrial
        for trial in range(0, max_ntrial):
            nnode = 2**(trial + 1)
            nnodes[trial] = nnode
            
            if quad_type == 'lg':
                [nodes, weights] = qd.lg_quad(nnode)
            elif quad_type == 'lgr':
                [nodes, weights] = qd.lgr_quad(nnode)
            elif quad_type == 'lgl':
                [nodes, weights] = qd.lgl_quad(nnode)
            elif quad_type == 'uni':
                [nodes, weights] = qd.uni_quad(nnode)
            else:
                msg = (
                    'ERROR: Test 2 recieved invalid quad_type. ' +
                    'Please use "lg", "lgr", "lgl", "uni".'
                )
                print(msg)
                quit()
                
            # Calculate f on nodes for quadrature
            f_nodes = f(nodes)
            weights = weights.reshape([1, nnode])
            f_nodes = f_nodes.reshape([nnode, 1])
            num_intg = weights @ f_nodes
            anl_intg = F(1) - F(-1)
            
            errs[trial] = np.abs(anl_intg - num_intg[0])
            
        # Plot errors
        fig, ax = plt.subplots()
        ax.plot(nnodes, errs,
                color = 'k',
                linestyle = '--')
        
        ax.set_xscale('log', base = 2)
        ax.set_yscale('log', base = 10)
        
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Error')
        
        title_str = ('1-D Function Integration via\n'
                     + '{} Quadrature').format(quad_type_str)
        ax.set_title(title_str)
        
        file_name = '{}_intg_acc_{}.png'.format(quad_type, func_num)
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(os.path.join(test_dir, file_name), dpi = 300)
        plt.close(fig)
