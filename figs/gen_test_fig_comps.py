import argparse
import json
import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
import os
import sys
from   scipy.integrate import quad, dblquad
from   time            import perf_counter

from   test_combos   import h_uni_ang, p_uni_ang, hp_uni_ang, \
    h_uni_spt, p_uni_spt, hp_uni_spt, \
    h_uni_all, p_uni_all, hp_uni_all, \
    h_amr_ang, p_amr_ang, hp_amr_ang, \
    h_amr_spt, p_amr_spt, hp_amr_spt, \
    h_amr_all, p_amr_all, hp_amr_all

def main():
    """
    Creates the compilation convegence plots
    """
    
    parser_desc = 'Determine which tests to run and where to put output.'
    parser = argparse.ArgumentParser(description = parser_desc)
    
    parser.add_argument('--dir', nargs = 1, default = 'figs',
                        required = False, help = 'Subdirectory to store output')
    help_str = 'Test Case Number - See Paper for Details'
    parser.add_argument('--test_num', nargs = 1, default = [1],
                        type = int, choices = [1, 2, 3, 4], required = False,
                        help = help_str)
    
    args = parser.parse_args()
    dir_name = args.dir
    test_num = args.test_num[0]
    
    figs_dir_name = 'test_{}_figs'.format(test_num)
    figs_dir = os.path.join(dir_name, figs_dir_name)
    os.makedirs(figs_dir, exist_ok = True)
    
    # Parameters for mesh, and plot functions
    if test_num == 1:
        combos = [
            h_uni_ang,
            p_uni_ang,
            h_amr_ang,
            hp_amr_ang
        ]
        
    elif test_num == 2:
        combos = [
            h_uni_ang,
            p_uni_ang,
            h_amr_ang,
            hp_amr_ang
        ]
        
    elif test_num == 3:
        combos = [
            hp_amr_spt,
            hp_amr_ang,
            hp_amr_all
        ]
        
    elif test_num == 4:
        combos = [
            hp_amr_spt,
            hp_amr_ang,
            hp_amr_all
        ]
        
    fig, ax = plt.subplots()
    
    ncombo = len(combos)
    
    colors = ['#000000', '#E69F00', '#56B4E9', '#009E73',
              '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
              '#882255']
    
    for cc in range(0, ncombo):
        combo      = combos[cc]
        combo_name = combo['short_name']
        combo_dir  = os.path.join(figs_dir, combo_name)
        
        ndofs_file_name = 'ndofs.txt'
        ndofs_file_path = os.path.join(combo_dir, ndofs_file_name)
        ndofs_file = open(ndofs_file_path, 'r')
        ndofs = json.load(ndofs_file)
        ndofs_file.close()
        
        errs_file_name = 'errs.txt'
        errs_file_path = os.path.join(combo_dir, errs_file_name)
        errs_file = open(errs_file_path, 'r')
        errs = json.load(errs_file)
        errs_file.close()
        
        ax.scatter(ndofs, errs,
                   label     = '{}'.format(combo_name),
                   color     = colors[cc]
                   )
        
        # Get best-fit line
        #[a, b] = np.polyfit(np.log10(ndofs), np.log10(errs), 1)
        #xx = np.logspace(np.log10(ndofs[0]), np.log10(ndofs[-1]))
        #yy = 10**b * xx**a
        #ax.plot(xx, yy,
        #        label = '{}: {:4.2f}'.format(combo_name, a),
        #        color = colors[cc],
        #        linestyle = '--'
        #        )
        
    ax.legend()
    
    ax.set_xscale('log', base = 10)
    ax.set_yscale('log', base = 10)
    
    ax.set_xlabel('Total Degrees of Freedom')
    if test_num == 1:
        ax.set_ylabel(r'$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u - u_{hp} \right)^2\,d\vec{x}\,d\hat{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\hat{s}}}$')
    elif test_num == 2:
        ax.set_ylabel(r'$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} - u_{hp} \right)^2\,d\vec{x}\,d\hat{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} \right)^2\,d\vec{x}\,d\hat{s}}}$')
    elif test_num == 3:
        ax.set_ylabel(r'$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} - u_{hp} \right)^2\,d\vec{x}\,d\hat{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} \right)^2\,d\vec{x}\,d\hat{s}}}$')
    elif test_num == 3:
        ax.set_ylabel(r'$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} - u_{hp} \right)^2\,d\vec{x}\,d\hat{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} \right)^2\,d\vec{x}\,d\hat{s}}}$')
    
    title_str = ( 'Convergence Rate' )
    ax.set_title(title_str)
    
    file_name = 'convergence_{}.png'.format(test_num)
    file_path = os.path.join(figs_dir, file_name)
    fig.set_size_inches(6.5, 6.5)
    plt.tight_layout()
    plt.savefig(file_path, dpi = 300)
    plt.close(fig)

if __name__ == '__main__':
    main()
