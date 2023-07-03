import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
from time import perf_counter
import os, sys


sys.path.append('../src')
from dg.mesh import Mesh
from dg.mesh.utils import plot_mesh, plot_mesh_p
from dg.matrix import get_intr_mask, split_matrix, merge_vectors
from dg.projection import Projection, to_projection
from dg.projection.utils import plot_xy, plot_xth, plot_yth, plot_xyth
import dg.quadrature as qd
from rt import rtdg
from amr import cell_jump_err, col_jump_err, rand_err, high_res_err, \
    low_res_err, nneg_err, ref_by_ind
from amr.utils import plot_error_indicator

from utils import print_msg

def main(dir_name = 'figs'):
    """
    Solves constructed ("manufactured") problems, with options for sub-problems
    and different types of refinement.
    """
    
    figs_dir = os.path.join(dir_name, 'test_2_figs')
    os.makedirs(figs_dir, exist_ok = True)
    
    # Test parameters:
    # Maximum number of DOFs
    max_ndof = 2**17
    # Maximum number of trials
    max_ntrial = 24
    # Minimum error before cut-off
    min_err = 10.**(-7)
    # Which combinations of Refinement Form, Refinement Type, and Refinement Kind
    combo_0 = {'full_name'  : 'Uniform Angular h-Refinement',
               'short_name' : 'h-uni-ang',
               'ref_type'   : 'uni',
               'ref_col'      : True,
               'col_ref_form' : 'h',
               'col_ref_kind' : 'ang',
               'col_ref_tol'  : 0.0,
               'ref_cell'      : False,
               'cell_ref_form' : None,
               'cell_ref_kind' : None,
               'cell_ref_tol'  : None}
    combo_1 = {'full_name'  : 'Uniform Angular p-Refinement',
               'short_name' : 'p-uni-ang',
               'ref_type'   : 'uni',
               'ref_col'      : True,
               'col_ref_form' : 'p',
               'col_ref_kind' : 'ang',
               'col_ref_tol'  : 0.0,
               'ref_cell'      : False,
               'cell_ref_form' : None,
               'cell_ref_kind' : None,
               'cell_ref_tol'  : None}
    combo_2 = {'full_name'  : 'Inhomogenous Isotropic Adaptive Angular h-Refinement',
               'short_name' : 'h-ii-amr-jmp-ang',
               'ref_type'   : 'amr-jmp-ang',
               'ref_col'      : True,
               'col_ref_form' : 'h',
               'col_ref_kind' : 'ang',
               'col_ref_tol'  : 0.85,
               'ref_cell'      : False,
               'cell_ref_form' : None,
               'cell_ref_kind' : None,
               'cell_ref_tol'  : None}
    combo_3 = {'full_name'  : 'Inhomogenous Isotropic Adaptive Angular hp-Refinement',
               'short_name' : 'hp-ii-amr-jmp-ang',
               'ref_type'   : 'amr-jmp-ang',
               'ref_col'      : True,
               'col_ref_form' : 'hp',
               'col_ref_kind' : 'ang',
               'col_ref_tol'  : 0.85,
               'ref_cell'      : False,
               'cell_ref_form' : None,
               'cell_ref_kind' : None,
               'cell_ref_tol'  : None}
    combo_4 = {'full_name'  : 'Inhomogenous Anisotropic Adaptive Angular h-Refinement',
               'short_name' : 'h-ia-amr-jmp-ang',
               'ref_type'   : 'amr-jmp-ang',
               'ref_col'      : False,
               'col_ref_form' : None,
               'col_ref_kind' : None,
               'col_ref_tol'  : None,
               'ref_cell'      : True,
               'cell_ref_form' : 'h',
               'cell_ref_kind' : 'ang',
               'cell_ref_tol'  : 0.85}
    combo_5 = {'full_name'  : 'Inhomogenous Anisotropic Adaptive Angular hp-Refinement',
               'short_name' : 'hp-ia-amr-jmp-ang',
               'ref_type'   : 'amr-jmp-ang',
               'ref_col'      : False,
               'col_ref_form' : None,
               'col_ref_kind' : None,
               'col_ref_tol'  : None,
               'ref_cell'      : True,
               'cell_ref_form' : 'hp',
               'cell_ref_kind' : 'ang',
               'cell_ref_tol'  : 0.85}
    combo_6 = {'full_name'  : 'Inhomogenous Anisotropic Adaptive Angular hp-Refinement',
               'short_name' : 'hp-ia-amr-nneg-ang',
               'ref_type'   : 'nneg',
               'ref_col'      : False,
               'col_ref_form' : None,
               'col_ref_kind' : None,
               'col_ref_tol'  : None,
               'ref_cell'      : True,
               'cell_ref_form' : 'hp',
               'cell_ref_kind' : 'ang',
               'cell_ref_tol'  : 0.0}
    combo_7 = {'full_name'  : 'Inhomogenous Anisotropic Adaptive Angular hp-Refinement',
               'short_name' : 'hp-ii-amr-nneg-ang',
               'ref_type'   : 'nneg',
               'ref_col'      : True,
               'col_ref_form' : 'hp',
               'col_ref_kind' : 'ang',
               'col_ref_tol'  : 0.0,
               'ref_cell'      : False,
               'cell_ref_form' : None,
               'cell_ref_kind' : None,
               'cell_ref_tol'  : None}
    
    combos = [
        combo_0
    ]
    
    # Output options
    do_plot_mesh        = False
    do_plot_mesh_p      = True
    do_plot_uh          = True
    do_plot_err_ind     = True
    do_plot_errs        = True
    
    combo_names = []
    combo_ndofs = {}
    combo_high_res_errs = {}
    
    perf_all_0 = perf_counter()
    msg = ( 'Generating test 2 figures...\n' )
    print_msg(msg)
    
    for combo in combos:
        combo_name = combo['short_name']
        combo_names += [combo_name]
        combo_dir = os.path.join(figs_dir, combo_name)
        os.makedirs(combo_dir, exist_ok = True)
        
        msg = ( 'Starting combination {}...\n'.format(combo['full_name']) )
        print_msg(msg)
        
        perf_combo_0 = perf_counter()
        perf_setup_0 = perf_counter()
        
        # Get the base mesh, manufactured solution
        [Lx, Ly]                   = [3., 2.]
        pbcs                       = [False, False]
        [ndof_x, ndof_y, ndof_th]  = [8, 8, 4]
        has_th                     = True
        
        mesh = Mesh(Ls     = [Lx, Ly],
                    pbcs   = pbcs,
                    ndofs  = [ndof_x, ndof_y, ndof_th],
                    has_th = has_th)
        
        for _ in range(0, 2):
            mesh.ref_mesh(kind = 'ang', form = 'h')
            
        for _ in range(0, 2):
            mesh.ref_mesh(kind = 'spt', form = 'h')
            
        # Randomly refine to start
        for _ in range(0, 0):
            kwargs = {'ref_col'      : True,
                      'col_ref_form' : 'hp',
                      'col_ref_kind' : 'spt',
                      'col_ref_tol'  : 0.5,
                      'ref_cell'      : True,
                      'cell_ref_form' : 'hp',
                      'cell_ref_kind' : 'ang',
                      'cell_ref_tol'  : 0.5}
            rand_err_ind = rand_err(mesh, **kwargs)
            
            mesh = ref_by_ind(mesh, rand_err_ind)
            
        # Test problem : No spatial dependence
        def kappa_x(x):
            return np.ones_like(x)
        def kappa_y(y):
            return np.ones_like(y)
        def kappa(x, y):
            return kappa_x(x) * kappa_y(y)
        def sigma(x, y):
            return 0.2 * kappa(x, y)
        g = 0.8
        def Phi_HG(Th):
            return (1. - g**2) / (1 + g**2 - 2. * g * np.cos(Th))**(3./2.)
        [norm, abserr] = quad(lambda Th : Phi_HG(Th), 0., 2. * np.pi)
        def Phi(th, phi):
            val = (1. - g**2) / (1 + g**2 - 2. * g * np.cos(th - phi))**(3./2.)
            return val / norm
        def f(x, y, th):
            return 0
        y_top = Ly
        def bcs(x, y, th):
            dth = 0.5
            ang_min = 3.0 * np.pi / 2.0 - dth / 2.0
            ang_max = 3.0 * np.pi / 2.0 + dth / 2.0
            if (y == y_top) and (ang_min <= th) and (th <= ang_max):
                return 10.0
            else:
                return 0.0
        dirac = [None, None, None]
            
        # Solve the manufactured problem over several trials
        ref_ndofs = []
        high_res_errs  = []
        
        ndof  = mesh.get_ndof()
        trial = 0
        err   = 1.
        
        perf_setup_f = perf_counter()
        perf_setup_diff = perf_setup_f - perf_setup_0
        msg = ( 'Combination {} setup complete!\n'.format(combo['full_name']) +
                12 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_setup_diff)
                )
        print_msg(msg)
        
        while (ndof < max_ndof) and (trial < max_ntrial) and (err > min_err):
            ndof = mesh.get_ndof()
            ref_ndofs += [ndof]
            
            perf_trial_0 = perf_counter()
            msg = '[Trial {}] Starting with {} of {} ndofs...\n'.format(trial, ndof, max_ndof)
            print_msg(msg)
            
            # Set up output directories
            trial_dir = os.path.join(combo_dir, 'trial_{}'.format(trial))
            os.makedirs(trial_dir, exist_ok = True)
            
            perf_0 = perf_counter()
            msg = ( '[Trial {}] Obtaining numerical solution...\n'.format(trial)
                   )
            print_msg(msg)
            
            uh_proj = rtdg(mesh, kappa, sigma, Phi, [bcs, dirac], f, verbose = True)
            
            perf_f = perf_counter()
            perf_diff = perf_f - perf_0
            msg = ( '[Trial {}] Numerical solution obtained!\n'.format(trial) +
                    22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                   )
            print_msg(msg)
            
            # Caluclate high-resolution error
            perf_0 = perf_counter()
            msg = ( '[Trial {}] Obtaining high-resolution error...\n'.format(trial)
                   )
            print_msg(msg)
            
            high_res_err_ind = high_res_err(mesh, uh_proj,
                                          kappa, sigma, Phi, [bcs, dirac], f,
                                          **combo)
            
            perf_f = perf_counter()
            perf_diff = perf_f - perf_0
            msg = ( '[Trial {}] High-resolution error obtained!\n'.format(trial) +
                    22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                   )
            print_msg(msg)

            if combo['ref_col']:
                err           = high_res_err_ind.col_max_err
                high_res_errs += [err]
            else: #if combo['ref_cell']
                err           = high_res_err_ind.cell_max_err
                high_res_errs += [err]
            
            if do_plot_mesh:
                perf_0 = perf_counter()
                msg = ( '[Trial {}] Plotting mesh...\n'.format(trial)
                       )
                print_msg(msg)
                
                file_name = 'mesh_3d_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_mesh(mesh      = mesh,
                          file_name = file_path,
                          plot_dim  = 3)
                
                file_name = 'mesh_2d_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_mesh(mesh        = mesh,
                          file_name   = file_path,
                          plot_dim    = 2,
                          label_cells = (trial <= 2))
                
                perf_f = perf_counter()
                perf_diff = perf_f - perf_0
                msg = ( '[Trial {}] Mesh plotted!\n'.format(trial) +
                        22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                       )
                print_msg(msg)
                
            if do_plot_mesh_p:
                perf_0 = perf_counter()
                msg = ( '[Trial {}] Plotting mesh polynomial degree...\n'.format(trial)
                       )
                print_msg(msg)
                
                file_name = 'mesh_3d_p_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_mesh_p(mesh        = mesh,
                            file_name   = file_path,
                            plot_dim    = 3)
                
                file_name = 'mesh_2d_p_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_mesh_p(mesh        = mesh,
                            file_name   = file_path,
                            plot_dim    = 2,
                            label_cells = (trial <= 3))
                
                perf_f = perf_counter()
                perf_diff = perf_f - perf_0
                msg = ( '[Trial {}] Mesh polynomial degree plotted!\n'.format(trial) +
                        22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                       )
                print_msg(msg)
                
            if do_plot_uh:
                perf_0 = perf_counter()
                msg = ( '[Trial {}] Plotting numerical solution...\n'.format(trial)
                       )
                print_msg(msg)
                
                file_name = 'uh_xy_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_xy(mesh, uh_proj, file_name = file_path)
                
                file_name = 'uh_xth_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_xth(mesh, uh_proj, file_name = file_path)
                
                file_name = 'uh_yth_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_yth(mesh, uh_proj, file_name = file_path)
                
                file_name = 'uh_xyth_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_xyth(mesh, uh_proj, file_name = file_path)
                
                perf_f = perf_counter()
                perf_diff = perf_f - perf_0
                msg = ( '[Trial {}] Numerical solution plotted!\n'.format(trial) +
                        22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                       )
                print_msg(msg)
                
                
            if do_plot_err_ind:
                    perf_0 = perf_counter()
                    msg = ( '[Trial {}] Plotting analytic error indicator...\n'.format(trial)
                           )
                    print_msg(msg)
                
                    file_name = 'high_res_err_ind.png'
                    file_path = os.path.join(trial_dir, file_name)
                    plot_error_indicator(mesh, high_res_err_ind, file_name = file_path)
                    
                    perf_f = perf_counter()
                    perf_diff = perf_f - perf_0
                    msg = ( '[Trial {}] Analytic error indicator plotted!\n'.format(trial) +
                            22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                           )
                    print_msg(msg)
                
            # Refine the mesh for the next trial
            ref_type = combo['ref_type']
            if ref_type == 'uni': # Uniform
                mesh.ref_mesh(kind = combo['col_ref_kind'],
                              form = combo['col_ref_form'])
            else:
                if ref_type == 'amr-jmp-ang': # Analytic angular error indicator
                    err_ind = cell_jump_err(mesh, uh_proj, **combo)
                elif ref_type == 'amr-jmp-spt': # Analytic angular error indicator
                    err_ind = col_jump_err(mesh, uh_proj, **combo)
                elif ref_type == 'nneg': # Non-negative error indicator
                    err_ind = nneg_err(mesh, uh_proj, **combo)
                    
                if do_plot_err_ind:
                    perf_0 = perf_counter()
                    msg = ( '[Trial {}] Plotting error indicator...\n'.format(trial)
                           )
                    print_msg(msg)
                
                    file_name = 'err_ind.png'
                    file_path = os.path.join(trial_dir, file_name)
                    plot_error_indicator(mesh, err_ind, file_name = file_path)
                    
                    perf_f = perf_counter()
                    perf_diff = perf_f - perf_0
                    msg = ( '[Trial {}] Error indicator plotted!\n'.format(trial) +
                            22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                           )
                    print_msg(msg)
                    
                mesh = ref_by_ind(mesh, err_ind)
                
            perf_trial_f    = perf_counter()
            perf_trial_diff = perf_trial_f - perf_trial_0
            msg = (
                '[Trial {}] Trial completed!\n'.format(trial) +
                22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_trial_diff)
            )
            print_msg(msg)
            
            trial += 1
            
        if do_plot_errs:
            fig, ax = plt.subplots()
            
            colors = ['#000000', '#E69F00', '#56B4E9', '#009E73',
                      '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
                      '#882255']
            
            ax.plot(ref_ndofs, high_res_errs,
                    label     = 'High-Resolution Error',
                    color     = colors[0],
                    linestyle = '--')
            
            ax.set_xscale('log', base = 2)
            ax.set_yscale('log', base = 10)
            
            errs = high_res_errs
            max_err = max(errs)
            min_err = min(errs)
            if np.log2(max_err) - np.log2(min_err) < 1:
                ymin = 2**(np.floor(np.log2(min_err)))
                ymax = 2**(np.ceil(np.log2(max_err)))
                ax.set_ylim([ymin, ymax])
                
            ax.set_xlabel('Total Degrees of Freedom')
            ax.set_ylabel('Error')
            
            ax.legend()
                
            title_str = ( '{} Convergence Rate'.format(combo['full_name']) )
            ax.set_title(title_str)
            
            file_name = 'convergence.png'
            file_path = os.path.join(combo_dir, file_name)
            fig.set_size_inches(6.5, 6.5)
            plt.savefig(file_path, dpi = 300)
            plt.close(fig)
            
            combo_ndofs[combo_name]    = ref_ndofs
            combo_high_res_errs[combo_name] = high_res_errs
            
        perf_combo_f = perf_counter()
        perf_combo_dt = perf_combo_f - perf_combo_0
        msg = (
            'Combination {} complete!\n'.format(combo['full_name']) +
            12 * ' ' + 'Time elapsed: {:08.3f} [s]\n'.format(perf_combo_dt)
        )
        print_msg(msg)
        
    if do_plot_errs:
        fig, ax = plt.subplots()
        
        ncombo = len(combos)
        
        colors = ['#000000', '#E69F00', '#56B4E9', '#009E73',
                  '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
                  '#882255']
        
        for cc in range(0, ncombo):
            combo_name = combo_names[cc]
            ax.plot(combo_ndofs[combo_name], combo_high_res_errs[combo_name],
                    label     = combo_name,
                    color     = colors[cc],
                    linestyle = '--')
            
        ax.legend()
        
        ax.set_xscale('log', base = 2)
        ax.set_yscale('log', base = 10)
        
        ax.set_xlabel('Total Degrees of Freedom')
        ax.set_ylabel('Error')
        
        title_str = ( 'Convergence Rate' )
        ax.set_title(title_str)
        
        file_name = 'convergence.png'
        file_path = os.path.join(figs_dir, file_name)
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(file_path, dpi = 300)
        plt.close(fig)
        
        perf_all_f = perf_counter()
        perf_all_dt = perf_all_f - perf_all_0
        msg = (
            'Test 1 figures generated!\n' +
            12 * ' ' + 'Time elapsed: {:08.3f} [s]\n'.format(perf_all_dt)
        )
        print_msg(msg)

if __name__ == '__main__':
    main()
