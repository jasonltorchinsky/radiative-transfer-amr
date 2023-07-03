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
from amr import anl_err, anl_err_ang, anl_err_spt, cell_jump_err, col_jump_err, \
    rand_err, high_res_err, low_res_err, nneg_err, ref_by_ind
from amr.utils import plot_error_indicator

from utils import print_msg

def main(dir_name = 'figs'):
    """
    Solves constructed ("manufactured") problems, with options for sub-problems
    and different types of refinement.
    """
    
    figs_dir = os.path.join(dir_name, 'test_1_figs')
    os.makedirs(figs_dir, exist_ok = True)
    
    # Test parameters:
    # Maximum number of DOFs
    max_ndof = 2**18
    # Maximum number of trials
    max_ntrial = 24
    # Minimum error before cut-off
    min_err = 10.**(-8)
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
    combo_2 = {'full_name'  : 'Inhomogenous Anisotropic Adaptive Angular h-Refinement',
               'short_name' : 'h-ia-amr-jmp-ang',
               'ref_type'   : 'amr-jmp-ang',
               'ref_col'      : False,
               'col_ref_form' : None,
               'col_ref_kind' : None,
               'col_ref_tol'  : None,
               'ref_cell'      : True,
               'cell_ref_form' : 'h',
               'cell_ref_kind' : 'ang',
               'cell_ref_tol'  : 0.90}
    combo_3 = {'full_name'  : 'Inhomogenous Anisotropic Adaptive Angular hp-Refinement',
               'short_name' : 'hp-ia-amr-jmp-ang',
               'ref_type'   : 'amr-jmp-ang',
               'ref_col'      : False,
               'col_ref_form' : None,
               'col_ref_kind' : None,
               'col_ref_tol'  : None,
               'ref_cell'      : True,
               'cell_ref_form' : 'hp',
               'cell_ref_kind' : 'ang',
               'cell_ref_tol'  : 0.90}
    
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
    do_plot_u           = True
    do_plot_diff        = True
    do_plot_err_ind     = True
    do_plot_errs        = True
    
    combo_names = []
    combo_ndofs = {}
    combo_anl_errs = {}
    
    perf_all_0 = perf_counter()
    msg = ( 'Generating test 1 figures...\n' )
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
            
        # Manufactured solution
        def X(x):
            return np.exp(-((1. / Lx) * (x - (Lx / 3.)))**2)
        def dXdx(x):
            return -(2. / Lx**2) * (x - (Lx / 3.)) * X(x)
        def Y(y):
            return np.exp(-4. * (Ly - y) / Ly)
        def dYdy(y):
            return (4. / Ly) * Y(y)
        def XY(x, y):
            return X(x) * Y(y)
        sth = 96.
        def Theta(th):
            return np.exp(-((sth / (2. * np.pi)) * (th - (7. * np.pi / 5.)))**2)
        def u(x, y, th):
            return XY(x, y) * Theta(th)
        
        def kappa_x(x):
            return np.exp(-((1. / Lx) * (x - (Lx / 2.)))**2)
        def kappa_y(y):
            return np.exp(-y / Ly)
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
            # Propagation part
            prop = (np.cos(th) * dXdx(x) * Y(y) + np.sin(th) * X(x) * dYdy(y)) * Theta(th)
            # Extinction part
            extn = kappa(x, y) * u(x, y, th)
            # Scattering part
            [Theta_scat, _] = quad(lambda phi: Phi(th, phi) * Theta(phi), 0., 2. * np.pi)
            scat =  sigma(x, y) * XY(x, y) * Theta_scat
            return prop + extn - scat
        def bcs(x, y, th):
            return u(x, y, th)
        dirac = [None, None, None]
        
        def u_intg_th(x, y, th0, th1):
            [Theta_intg, _] = quad(lambda th: Theta(th), th0, th1)
            return XY(x, y) * Theta_intg
        
        def u_intg_xy(x0, x1, y0, y1, th):
            [XY_intg, _] = dblquad(lambda x, y: XY(x, y), x0, x1, y0, y1)
            return XY_intg * Theta(th)
        
        # Perform some uniform (angular or spatial) h-refinements to start
        for _ in range(0, 0):
            mesh.ref_mesh(kind = 'all', form = 'h')
            
        # Solve the manufactured problem over several trials
        ref_ndofs = []
        anl_err_dofs = []
        anl_errs  = []
        
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
            calc_anl_err = False
            
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
            
            # Caluclate analytic error
            if (trial == 0) or (np.isclose(np.log2(ndof), np.round(np.log2(ndof), decimals = 1), atol = 1.e-4)):
                perf_0 = perf_counter()
                msg = ( '[Trial {}] Obtaining analytic error...\n'.format(trial)
                       )
                print_msg(msg)
                
                anl_err_ind = anl_err(mesh, uh_proj, u, ref_col = False, ref_cell = True)
                
                perf_f = perf_counter()
                perf_diff = perf_f - perf_0
                msg = ( '[Trial {}] Analytic error obtained!\n'.format(trial) +
                        22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                       )
                print_msg(msg)
                
                err           = anl_err_ind.cell_max_err
                anl_err_dofs += [ndof]
                anl_errs     += [err]
                calc_anl_err  = True
            
            if do_plot_mesh and (trial%3 == 0):
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
                
            if do_plot_mesh_p and (trial%3 == 0):
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
                
            if do_plot_uh and (trial%3 == 0):
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
                
            if do_plot_u and (trial%1 == 0):
                perf_0 = perf_counter()
                msg = ( '[Trial {}] Plotting analytic solution...\n'.format(trial)
                       )
                print_msg(msg)
                
                u_proj = Projection(mesh, u)

                file_name = 'u_th_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_th(mesh, u_proj, file_name = file_path)
            
                file_name = 'u_xy_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_xy(mesh, u_proj, file_name = file_path)
                
                file_name = 'u_xth_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_xth(mesh, u_proj, file_name = file_path)
                
                file_name = 'u_yth_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_yth(mesh, u_proj, file_name = file_path)
                
                file_name = 'u_xyth_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_xyth(mesh, u_proj, file_name = file_path)
                
                perf_f = perf_counter()
                perf_diff = perf_f - perf_0
                msg = ( '[Trial {}] Analytic solution plotted!\n'.format(trial) +
                        22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                       )
                print_msg(msg)
                
            if do_plot_diff and (trial%3 == 0):
                perf_0 = perf_counter()
                msg = ( '[Trial {}] Plotting difference in solutions...\n'.format(trial)
                       )
                print_msg(msg)
                
                u_vec     = u_proj.to_vector()
                uh_vec    = uh_proj.to_vector()
                diff_vec  = u_vec - uh_vec
                diff_proj = to_projection(mesh, diff_vec)

                file_name = 'diff_th_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_th(mesh, diff_proj, file_name = file_path)
                
                file_name = 'diff_xy_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_xy(mesh, diff_proj, file_name = file_path,
                        cmap = 'bwr', scale = 'diff')
                
                file_name = 'diff_xth_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_xth(mesh, diff_proj, file_name = file_path,
                         cmap = 'bwr', scale = 'diff')
                
                file_name = 'diff_yth_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_yth(mesh, diff_proj, file_name = file_path,
                         cmap = 'bwr', scale = 'diff')
                
                file_name = 'diff_xyth_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_xyth(mesh, diff_proj, file_name = file_path,
                          cmap = 'bwr', scale = 'diff')
                
                perf_f = perf_counter()
                perf_diff = perf_f - perf_0
                msg = ( '[Trial {}] Difference in solutions plotted!\n'.format(trial) +
                        22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                       )
                print_msg(msg)
                
            if do_plot_err_ind and (trial%3 == 0) and calc_anl_err:
                    perf_0 = perf_counter()
                    msg = ( '[Trial {}] Plotting analytic error indicator...\n'.format(trial)
                           )
                    print_msg(msg)
                
                    file_name = 'anl_err_ind.png'
                    file_path = os.path.join(trial_dir, file_name)
                    plot_error_indicator(mesh, anl_err_ind, file_name = file_path)
                    
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
                if ref_type == 'amr-anl-ang': # Analytic angular error indicator
                    err_ind = anl_err_ang(mesh, uh_proj, u_intg_xy, **combo)
                elif ref_type == 'amr-anl-spt': # Analytic angular error indicator
                    err_ind = anl_err_spt(mesh, uh_proj, u_intg_th, **combo)
                elif ref_type == 'amr-jmp-ang': # Analytic angular error indicator
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
            
            ax.plot(anl_err_dofs, anl_errs,
                    label     = 'Analytic Error',
                    color     = colors[0],
                    linestyle = '--')
            
            ax.set_xscale('log', base = 2)
            ax.set_yscale('log', base = 10)
            
            errs = anl_errs
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
            
            combo_ndofs[combo_name]    = anl_err_dofs
            combo_anl_errs[combo_name] = anl_errs
            
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
            ax.plot(combo_ndofs[combo_name], combo_anl_errs[combo_name],
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
