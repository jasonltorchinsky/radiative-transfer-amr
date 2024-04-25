"""
Generates compilation figures for a given test case.
"""

# Standard Library Imports
import argparse
import json
import os
import sys
from   time            import perf_counter

# Third-Party Library Imports
import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
from   scipy.integrate import quad, dblquad

# Local Library Imports
import test_temp as test

def main():
    """
    Creates the compilation convegence plots
    """

    # Get local copy of variables from test.py - which is a temporary copy of a
    # test_n.py for some n
    max_ndof   = test.max_ndof
    max_ntrial = test.max_ntrial
    min_err    = test.min_err
    max_mem    = test.max_mem
    [Lx, Ly]   = [test.Lx, test.Ly]
    combos     = test.combos
    kappa      = test.kappa
    sigma      = test.sigma
    Phi        = test.Phi
    f          = test.f
    bcs_dirac  = test.bcs_dirac
    u          = test.u
    
    parser_desc = 'Determine which tests to run and where to put output.'
    parser = argparse.ArgumentParser(description = parser_desc)
    
    parser.add_argument('--dir', nargs = 1, default = 'figs',
                        required = False, help = 'Subdirectory to store output')
    help_str = 'Test Case Number - See Paper for Details'
    parser.add_argument('--test_num', nargs = 1, default = [1],
                        type = int, required = False,
                        help = help_str)
    
    args = parser.parse_args()
    dir_name = args.dir
    test_num = args.test_num[0]
    
    figs_dir_name = 'test_{}_figs'.format(test_num)
    figs_dir = os.path.join(dir_name, figs_dir_name)
    os.makedirs(figs_dir, exist_ok = True)
    
    # Parameters for mesh, and plot functions
    if test_num in [1, 2]:
        combo_long_names = [
            r'$h$-Unif. Ang.',
            r'$p$-Unif. Ang.',
            r'$h$-Adap. Ang.',
            r'$hp$-Adap. Ang.'
        ]
    else:
        combo_long_names = [
            r'$hp$-Adap. Spt. with $p$-Unif. Ang.',
            r'$hp$-Adap. Ang. with $p$-Unif. Spt.',
            r'$hp$-Adap. Ang. with $hp$-Adap. Spt.'
        ]

    fig, ax = plt.subplots()
    
    ncombo = len(combos)
    
    colors = ['#000000', '#E69F00', '#56B4E9', '#009E73',
              '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
              '#882255']
    mstyles = ['.', 'v', 's', '*', 'h', 'D']
    
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
        
        ax.plot(ndofs, errs,
                label     = '{}'.format(combo_long_names[cc]),
                color     = colors[cc%len(colors)],
                marker    = mstyles[cc%len(mstyles)],
                linestyle = ':'
                )
        
    ax.legend()
    
    #ax.set_xscale('log', base = 10)
    ax.set_yscale('log', base = 10)
    
    ax.set_xlabel('Total Degrees of Freedom')
    if test_num == 1:
        ax.set_ylabel(r'Error := $\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u - u_{hp} \right)^2\,d\vec{x}\,d\vec{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\vec{s}}}$')
    else:
        ax.set_ylabel(r'Error := $\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} - u_{hp} \right)^2\,d\vec{x}\,d\vec{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} \right)^2\,d\vec{x}\,d\vec{s}}}$')
    
    title_str = ( 'Convergence Test' )
    ax.set_title(title_str)
    
    file_name = 'dof_convergence_{}.png'.format(test_num)
    file_path = os.path.join(figs_dir, file_name)
    fig.set_size_inches(6.5, 6.5)
    plt.tight_layout()
    plt.savefig(file_path, dpi = 300)
    plt.close(fig)

    fig, ax = plt.subplots()
    
    for cc in range(0, ncombo):
        combo      = combos[cc]
        combo_name = combo['short_name']
        combo_dir  = os.path.join(figs_dir, combo_name)

        nnz_file_name = 'nnz.txt'
        nnz_file_path = os.path.join(combo_dir, nnz_file_name)
        nnz_file = open(nnz_file_path, 'r')
        nnz = json.load(nnz_file)
        nnz_file.close()
        
        errs_file_name = 'errs.txt'
        errs_file_path = os.path.join(combo_dir, errs_file_name)
        errs_file = open(errs_file_path, 'r')
        errs = json.load(errs_file)
        errs_file.close()
        
        ax.plot(nnz, errs,
                label     = '{}'.format(combo_long_names[cc]),
                color     = colors[cc%len(colors)],
                marker    = mstyles[cc%len(mstyles)],
                linestyle = ':'
                )
        
    ax.legend()
    
    #ax.set_xscale('log', base = 10)
    ax.set_yscale('log', base = 10)
    
    ax.set_xlabel('Non-Zeros in System Matrix')
    if test_num == 1:
        ax.set_ylabel(r'Error := $\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u - u_{hp} \right)^2\,d\vec{x}\,d\vec{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\vec{s}}}$')
    else:
        ax.set_ylabel(r'Error := $\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} - u_{hp} \right)^2\,d\vec{x}\,d\vec{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} \right)^2\,d\vec{x}\,d\vec{s}}}$')
    
    title_str = ( 'Convergence Test' )
    ax.set_title(title_str)
    
    file_name = 'nnz_convergence_{}.png'.format(test_num)
    file_path = os.path.join(figs_dir, file_name)
    fig.set_size_inches(6.5, 6.5)
    plt.tight_layout()
    plt.savefig(file_path, dpi = 300)
    plt.close(fig)

if __name__ == '__main__':
    main()
