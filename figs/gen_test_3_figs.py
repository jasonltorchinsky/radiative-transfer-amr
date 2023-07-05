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
from dg.projection.utils import plot_xy, plot_xth, plot_yth, plot_xyth, plot_th
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
    
    figs_dir = os.path.join(dir_name, 'test_3_figs')
    os.makedirs(figs_dir, exist_ok = True)
    
    # Test parameters:
    # Maximum number of DOFs
    max_ndof = 2**18
    # Maximum number of trials
    max_ntrial = 64
    # Minimum error before cut-off
    min_err = 1.e-10
    # Which combinations of Refinement Form, Refinement Type, and Refinement Kind
    combo_0 = {'full_name'  : 'Uniform Angular h-Refinement',
               'short_name' : 'h-uni-all'}
    combo_1 = {'full_name'  : 'Uniform Angular p-Refinement',
               'short_name' : 'p-uni-all'}
    combo_2 = {'full_name'  : 'Inhomogenous Anisotropic Adaptive Angular h-Refinement',
               'short_name' : 'h-amr-all',}
    combo_3 = {'full_name'  : 'Inhomogenous Anisotropic Adaptive Angular hp-Refinement',
               'short_name' : 'hp-amr-all'}
    
    combos = [
        combo_0,
        combo_1,
        combo_2,
        combo_3
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
    msg = ( 'Generating test 3 figures...\n' )
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
        pbcs                       = [True, False]
        [ndof_x, ndof_y, ndof_th]  = [3, 3, 3]
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
            
        # Test problem : Cloud layer with Rayleigh scattering
        def kappa_x(x):
            return (1. / 32.) * (np.sin(2. * np.pi * x / Lx))**2 + 0.1
        def kappa(x, y):
            term_0 = (-1. / ((2. * (Ly * kappa_x(x))**2))) * (y - (Ly / 2.))**2
            return 0.45 * (np.sign(np.exp(term_0) - 0.5) + 1.) + 0.1
        def sigma(x, y):
            return 0.9 * kappa(x, y)
        def Phi(th, phi):
            return (1. / (3. * np.pi)) * (1. + (np.cos(th - phi))**2)
        def f(x, y, th):
            return 0
        y_top = Ly
        def bcs(x, y, th):
            sth = 96.
            if (y == y_top):
                return np.exp(-((sth / (2. * np.pi)) * (th - (3. * np.pi / 2.)))**2)
            else:
                return 0
        dirac = [None, None, None]

        # Plot extinction coefficient, scattering coefficient, and scattering phase function
        xx = np.linspace(0, Lx, num = 1000).reshape([1, 1000])
        yy = np.linspace(0, Ly, num = 1000).reshape([1000, 1])
        [XX, YY] = np.meshgrid(xx, yy)

        th = np.linspace(0, 2. * np.pi, num = 360)

        kappa_c = kappa(xx, yy)
        sigma_c = sigma(xx, yy)
        [vmin, vmax] = [0., max(np.amax(kappa_c), np.amax(sigma_c))]
        rr = Phi(0, th)
        
        ## kappa
        fig, ax = plt.subplots()
        kappa_plot = ax.contourf(XX, YY, kappa_c, vmin = vmin, vmax = vmax)
        ax.set_xlim([0, Lx])
        ax.set_ylim([0, Ly])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(kappa_plot, ax = ax)
        file_name = 'kappa.png'
        file_path = os.path.join(combo_dir, file_name)
        plt.tight_layout()
        plt.savefig(file_path, dpi = 300)
        plt.close(fig)

        ## sigma
        fig, ax = plt.subplots()
        sigma_plot = ax.contourf(XX, YY, sigma_c, vmin = vmin, vmax = vmax)
        ax.set_xlim([0, Lx])
        ax.set_ylim([0, Ly])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(sigma_plot, ax = ax)
        file_name = 'sigma.png'
        file_path = os.path.join(combo_dir, file_name)
        plt.tight_layout()
        plt.savefig(file_path, dpi = 300)
        plt.close(fig)

        ## Phi
        max_r = max(rr)
        ntick = 2
        r_ticks = np.linspace(max_r / ntick, max_r, ntick)
        r_tick_labels = ['{:3.2f}'.format(r_tick) for r_tick in r_ticks]
        th_ticks = np.linspace(0, 2. * np.pi, num = 8, endpoint = False)
        th_tick_labels = [r'${:3.2f} \pi$'.format(th_tick/np.pi) for th_tick in th_ticks]
        fig, ax = plt.subplots(subplot_kw = {'projection': 'polar'})
        Phi_plot = ax.plot(th, rr, color = 'black')
        ax.set_rlim([0, max_r])
        ax.set_rticks(r_ticks, r_tick_labels)
        ax.set_xlabel(r"$\theta - \theta'$")
        ax.set_xticks(th_ticks, th_tick_labels)
        file_name = 'Phi.png'
        file_path = os.path.join(combo_dir, file_name)
        plt.tight_layout()
        plt.savefig(file_path, dpi = 300)
        plt.close(fig)
        
        
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
        
        while (ndof < max_ndof):# and (trial < max_ntrial) and (err > min_err):
            ndof = mesh.get_ndof()
            ref_ndofs += [ndof]
            
            perf_trial_0 = perf_counter()
            msg = '[Trial {}] Starting with {} of {} ndofs and error {:.4E}...\n'.format(trial, ndof, max_ndof, err)
            print_msg(msg)
            
            # Set up output directories
            trial_dir = os.path.join(combo_dir, 'trial_{}'.format(trial))
            os.makedirs(trial_dir, exist_ok = True)
            
            perf_0 = perf_counter()
            msg = ( '[Trial {}] Obtaining numerical solution...\n'.format(trial)
                   )
            print_msg(msg)
            
            [uh_proj, info] = rtdg(mesh, kappa, sigma, Phi, [bcs, dirac], f,
                                   verbose = True, solver = 'gmres',
                                   precondition = True)
            
            perf_f = perf_counter()
            perf_diff = perf_f - perf_0
            msg = ( '[Trial {}] Numerical solution obtained!\n'.format(trial) +
                    22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                   )
            print_msg(msg)
            
            # Caluclate high-resolution error
            if (trial%1 == 0):
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

                file_name = 'uh_th_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_th(mesh, uh_proj, file_name = file_path)
                
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
                
                
            if False:#do_plot_err_ind:
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
            if combo['short_name'] == 'h-uni-all':
                mesh.ref_mesh(kind = 'all', form = 'h')
            elif combo['short_name'] == 'p-uni-all':
                mesh.ref_mesh(kind = 'all', form = 'p')
            elif combo['short_name'] == 'h-amr-all':
                kwargs_ang = {'ref_col'      : False,
                              'col_ref_form' : None,
                              'col_ref_kind' : None,
                              'col_ref_tol'  : None,
                              'ref_cell'      : True,
                              'cell_ref_form' : 'h',
                              'cell_ref_kind' : 'ang',
                              'cell_ref_tol'  : 0.9}
                err_ind_ang = cell_jump_err(mesh, uh_proj, **kwargs_ang)
                
                kwargs_spt = {'ref_col'      : True,
                              'col_ref_form' : 'h',
                              'col_ref_kind' : 'spt',
                              'col_ref_tol'  : 0.85,
                              'ref_cell'      : False,
                              'cell_ref_form' : None,
                              'cell_ref_kind' : None,
                              'cell_ref_tol'  : None}
                err_ind_spt = col_jump_err( mesh, uh_proj, **combo)
                
                if do_plot_err_ind:
                    perf_0 = perf_counter()
                    msg = ( '[Trial {}] Plotting error indicators...\n'.format(trial)
                           )
                    print_msg(msg)
                    
                    file_name = 'err_ind_ang.png'
                    file_path = os.path.join(trial_dir, file_name)
                    plot_error_indicator(mesh, err_ind_ang, file_name = file_path)
                    
                    file_name = 'err_ind_spt.png'
                    file_path = os.path.join(trial_dir, file_name)
                    plot_error_indicator(mesh, err_ind_spt, file_name = file_path)
                    
                    perf_f = perf_counter()
                    perf_diff = perf_f - perf_0
                    msg = ( '[Trial {}] Error indicator plotted!\n'.format(trial) +
                            22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                           )
                    print_msg(msg)
                    
                mesh = ref_by_ind(mesh, err_ind_ang)
                mesh = ref_by_ind(mesh, err_ind_spt)
                
            elif combo['short_name'] == 'hp-amr-all':
                kwargs_ang = {'ref_col'      : False,
                              'col_ref_form' : None,
                              'col_ref_kind' : None,
                              'col_ref_tol'  : None,
                              'ref_cell'      : True,
                              'cell_ref_form' : 'hp',
                              'cell_ref_kind' : 'ang',
                              'cell_ref_tol'  : 0.9}
                err_ind_ang = cell_jump_err(mesh, uh_proj, **kwargs_ang)
                
                kwargs_spt = {'ref_col'      : True,
                              'col_ref_form' : 'hp',
                              'col_ref_kind' : 'spt',
                              'col_ref_tol'  : 0.85,
                              'ref_cell'      : False,
                              'cell_ref_form' : None,
                              'cell_ref_kind' : None,
                              'cell_ref_tol'  : None}
                err_ind_spt = col_jump_err( mesh, uh_proj, **combo)
                
                if do_plot_err_ind:
                    perf_0 = perf_counter()
                    msg = ( '[Trial {}] Plotting error indicators...\n'.format(trial)
                           )
                    print_msg(msg)
                    
                    file_name = 'err_ind_ang.png'
                    file_path = os.path.join(trial_dir, file_name)
                    plot_error_indicator(mesh, err_ind_ang, file_name = file_path)
                    
                    file_name = 'err_ind_spt.png'
                    file_path = os.path.join(trial_dir, file_name)
                    plot_error_indicator(mesh, err_ind_spt, file_name = file_path)
                    
                    perf_f = perf_counter()
                    perf_diff = perf_f - perf_0
                    msg = ( '[Trial {}] Error indicator plotted!\n'.format(trial) +
                            22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                           )
                    print_msg(msg)
                    
                mesh = ref_by_ind(mesh, err_ind_ang)
                mesh = ref_by_ind(mesh, err_ind_spt)
                
                
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
