import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append('../src')
import dg.quadrature as qd

def main(dir_name = 'figs'):
    """
    Generats plots of the quadrature nodes and basis functions.
    """

    figs_dir = os.path.join(dir_name, 'basis_figs')
    os.makedirs(figs_dir, exist_ok = True)

    for quad_type in ['lg', 'lgl']:

        nnode = 5
        
        if quad_type == 'lg':
            [nodes, weights] = qd.lg_quad(nnode)
            quad_type_str = 'Legendre-Gauss'
        elif quad_type == 'lgr':
            [nodes, weights] = qd.lgr_quad(nnode)
            quad_type_str = 'Legendre-Gauss-Radau'
        elif quad_type == 'lgl':
            [nodes, weights] = qd.lgl_quad(nnode)
            quad_type_str = 'Legendre-Gauss-Lobatto'
        elif quad_type == 'uni':
            [nodes, weights] = qd.uni_quad(nnode)
            quad_type_str = 'Uniform'
        else:
            msg = (
                'ERROR: Test 0 recieved invalid quad_type. ' +
                'Please use "lg", "lgr", "lgl", "uni".'
            )
            print(msg)
            quit()
            
        # Calculate basis functions for plotting
        nx = 500
        xx = np.linspace(-1., 1., nx)
        basis_funcs = np.zeros([nnode, nx])
        for ii in range(0, nnode):
            for x_idx in range(0, nx):
                basis_funcs[ii, x_idx] = qd.lag_eval(nodes, ii, xx[x_idx])
        
        fig, ax = plt.subplots()
        
        
        # Plot basis functions
        colors = ['#E69F00', '#56B4E9', '#009E73', '#0072B2',
                  '#D55E00', '#CC79A7']
        for ii in range(0, nnode):
            lbl = 'Basis Function {}'.format(ii)
            ax.plot(xx, basis_funcs[ii, :], label = lbl,
                    linestyle = '-',
                    color = colors[ii])
        
        # Plot quadrature nodes
        yy = np.zeros_like(nodes)
        ax.scatter(nodes, yy,
                   color = 'k',
                   zorder = 3)
        
        ax.set_xlim([-1.1, 1.1])
        
        
        title_str = (
            '{} Nodes and Basis Functions'
        ).format(quad_type_str)
        ax.set_title(title_str)
        
        file_name = '{}_basis.png'.format(quad_type)
        file_path = os.path.join(figs_dir, file_name)
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(file_path, dpi = 300, bbox_inches = 'tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
