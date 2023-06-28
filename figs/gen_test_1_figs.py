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
    # AMR Refinement Tolerance
    tol_spt = 0.50
    tol_ang = 0.50
    # Maximum number of DOFs
    max_ndof = 2**16
    # Maximum number of trials
    max_ntrial = 1
    # Which combinations of Refinement Form, Refinement Type, and Refinement Kind
    combos = [
        ['hp',  'rng',     'spt']
    ]
    
    # Test Output Parameters
    do_plot_mesh        = False
    do_plot_mesh_p      = True
    do_plot_matrix      = False
    do_plot_uh          = False
    do_plot_u           = False
    do_plot_diff        = False
    do_plot_anl_err_ind = False
    do_plot_sol_vecs    = False
    do_calc_hi_res_err  = False
    do_calc_low_res_err = False
    do_plot_errs        = False
    
    combo_ndofs = {}
    combo_anl_errs = {}
    combo_hr_errs  = {}
    
    for combo in combos:
        [ref_form, ref_type, ref_kind] = combo
        combo_str = '{}-{}-{}'.format(ref_form, ref_type, ref_kind)
        combo_dir = os.path.join(figs_dir, combo_str)
        os.makedirs(combo_dir, exist_ok = True)
        
        msg = ( 'Starting combination {}...\n'.format(combo_str) )
        print_msg(msg)

        perf_combo_0 = perf_counter()
        perf_setup_0 = perf_counter()
        
        # Get the base mesh, manufactured solution
        [Lx, Ly]                   = [3., 2.]
        pbcs                       = [True, False]
        [ndof_x, ndof_y, ndof_th]  = [3, 3, 5]
        has_th                     = True
        
        mesh = Mesh(Ls     = [Lx, Ly],
                    pbcs   = pbcs,
                    ndofs  = [ndof_x, ndof_y, ndof_th],
                    has_th = has_th)
        
        for _ in range(0, 3):
            mesh.ref_mesh(kind = 'ang', form = 'h')

        for _ in range(0, 2):
            mesh.ref_mesh(kind = 'spt', form = 'h')
            
        # Randomly refine to start
        for _ in range(0, 2):
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
            
        # Manufactured solution from Shukai's paper
        def XY(x, y):
            return 1. + np.cos(2. * np.pi * x / Lx) * np.sin(np.pi * y / Ly)
        def Theta(th):
            val = np.piecewise(th,
                    [(0 <= th) * (th <= np.pi / 2.),
                     (3. * np.pi / 2. <= th) * (th <= 2. * np.pi),
                     (np.pi / 2. < th) * (th < 3. * np.pi / 2.)],
                        [lambda th: np.sin((np.pi / 2.) * (1. - 2. * th / np.pi)),
                         lambda th: ((2. / np.pi) * (th - 3. * np.pi / 2.))**3,
                         0.])
            return val
        def u(x, y, th):
            return XY(x, y) * Theta(th)
        
        def kappa(x, y):
            return 2.
        def sigma(x, y):
            return kappa(x, y) * 0.5 / (2. * np.pi)
        def Phi(th, phi):
            return (1 / (3. * np.pi)) * (1. + (np.cos(th - phi)**2))
        def f(x, y, th):
            # Propagation part
            prop = (- (2. * np.pi / Lx) * np.cos(th) * np.sin(2. * np.pi * x / Lx) * np.sin(np.pi * y / Ly) \
                + (np.pi / Ly) * np.sin(th) * np.cos(2. * np.pi * x / Lx) * np.cos(np.pi * y / Ly)) \
                * Theta(th)
            # Extinction part
            extn = kappa(x, y) * u(x, y, th)
            # Scattering part
            [Theta_scat, _] = quad(lambda phi: Phi(th, phi) * Theta(phi), 0., 2. * np.pi)
            scat =  sigma(x, y) * XY(x, y) * Theta_scat
            
            return (1. / (kappa(x, y) - 2 * np.pi * sigma(x, y))) * (prop + extn - scat)
        def bcs(x, y, th):
            return u(x, y, th)
        dirac = [None, Ly, None]
        
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
        anl_errs  = []
        anl_spt_errs = []
        anl_ang_errs = []
        jmp_spt_errs = []
        jmp_ang_errs = []
        hr_errs   = []
        lr_errs   = []

        ndof  = mesh.get_ndof()
        trial = 0

        perf_setup_f = perf_counter()
        perf_setup_diff = perf_setup_f - perf_setup_0
        msg = ( 'Combination {} setup complete!\n'.format(combo_str) +
                12 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_setup_diff)
                )
        print_msg(msg)
        
        while (ndof < max_ndof) and (trial < max_ntrial):
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
            
            #uh_proj = rtdg(mesh, kappa, sigma, Phi, [bcs, dirac], f)
            def zero(x, y, th):
                return 0
            uh_proj = Projection(mesh, zero)
            
            perf_f = perf_counter()
            perf_diff = perf_f - perf_0
            msg = ( '[Trial {}] Numerical solution obtained!\n'.format(trial) +
                    22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                   )
            print_msg(msg)

            # Caluclate error
            ## Analytic error
            if ref_kind == 'spt':
                kwargs = {'ref_col'      : True,
                          'col_ref_form' : ref_form,
                          'col_ref_kind' : 'spt',
                          'col_ref_tol'  : tol_spt,
                          'ref_cell'      : False}
            elif ref_kind == 'ang':
                kwargs = {'ref_col'      : False,
                          'ref_cell'      : True,
                          'cell_ref_form' : ref_form,
                          'cell_ref_kind' : 'ang',
                          'cell_ref_tol'  : tol_ang}
            elif ref_kind == 'all':
                kwargs = {'ref_col'      : True,
                          'col_ref_form' : ref_form,
                          'col_ref_kind' : 'spt',
                          'col_ref_tol'  : tol_spt,
                          'ref_cell'      : True,
                          'cell_ref_form' : ref_form,
                          'cell_ref_kind' : 'ang',
                          'cell_ref_tol'  : tol_ang}
            else:
                kwargs = {}
                
            perf_0 = perf_counter()
            msg = ( '[Trial {}] Obtaining analytic error...\n'.format(trial)
                   )
            print_msg(msg)
            
            anl_err_ind = anl_err(mesh, uh_proj, u, **kwargs)
            
            perf_f = perf_counter()
            perf_diff = perf_f - perf_0
            msg = ( '[Trial {}] Analytic error obtained!\n'.format(trial) +
                    22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                   )
            print_msg(msg)
            
            if ref_kind == 'spt':
                anl_errs += [anl_err_ind.col_max_err]
            elif ref_kind == 'ang':
                anl_errs += [anl_err_ind.cell_max_err]
            elif ref_kind == 'all':
                anl_errs += [max(anl_err_ind.col_max_err, anl_err_ind.cell_max_err)]
            else:
                anl_errs += [anl_err_ind.max_col_err]
                
            ## Analytic angularly-integrated error
            kwargs = {'ref_col'      : True,
                      'col_ref_form' : ref_form,
                      'col_ref_kind' : 'spt',
                      'col_ref_tol'  : tol_spt,
                      'ref_cell'      : False}
            
            perf_0 = perf_counter()
            msg = ( '[Trial {}] Obtaining analytic spatial error...\n'.format(trial)
                   )
            print_msg(msg)
            
            anl_err_ind_spt = anl_err_spt(mesh, uh_proj, u_intg_th, **kwargs)
            
            perf_f = perf_counter()
            perf_diff = perf_f - perf_0
            msg = ( '[Trial {}] Analytic spatial error obtained!\n'.format(trial) +
                    22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                   )
            print_msg(msg)
            
            anl_spt_errs += [anl_err_ind_spt.col_max_err]
            
            ## Analytic spatially-integrate error
            kwargs = {'ref_col'      : False,
                      'ref_cell'      : True,
                      'cell_ref_form' : ref_form,
                      'cell_ref_kind' : 'ang',
                      'cell_ref_tol'  : tol_ang}

            perf_0 = perf_counter()
            msg = ( '[Trial {}] Obtaining analytic angular error...\n'.format(trial)
                   )
            print_msg(msg)
            
            anl_err_ind_ang = anl_err_ang(mesh, uh_proj, u_intg_xy, **kwargs)
            
            perf_f = perf_counter()
            perf_diff = perf_f - perf_0
            msg = ( '[Trial {}] Analytic angular error obtained!\n'.format(trial) +
                    '             Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                   )
            print_msg(msg)
            
            anl_ang_errs += [anl_err_ind_ang.cell_max_err]
            
            ## Jump error
            kwargs = {'ref_col'      : True,
                      'col_ref_form' : ref_form,
                      'col_ref_kind' : 'spt',
                      'col_ref_tol'  : tol_spt,
                      'ref_cell'      : False}

            perf_0 = perf_counter()
            msg = ( '[Trial {}] Obtaining spatial jump error...\n'.format(trial)
                   )
            print_msg(msg)
            
            jmp_err_ind_spt = col_jump_err(mesh, uh_proj, **kwargs)
            
            perf_f = perf_counter()
            perf_diff = perf_f - perf_0
            msg = ( '[Trial {}] Spatial jump error obtained!\n'.format(trial) +
                    22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                   )
            print_msg(msg)
            
            jmp_spt_errs += [jmp_err_ind_spt.col_max_err]
            
            kwargs = {'ref_col'      : False,
                      'ref_cell'      : True,
                      'cell_ref_form' : ref_form,
                      'cell_ref_kind' : 'ang',
                      'cell_ref_tol'  : tol_ang}
            
            perf_0 = perf_counter()
            msg = ( '[Trial {}] Obtaining angular jump error...\n'.format(trial)
                   )
            print_msg(msg)
            
            jmp_err_ind_ang = cell_jump_err(mesh, uh_proj, **kwargs)
            
            perf_f = perf_counter()
            perf_diff = perf_f - perf_0
            msg = ( '[Trial {}] Spatial angular error obtained!\n'.format(trial) +
                    22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                   )
            print_msg(msg)
            
            jmp_ang_errs += [jmp_err_ind_ang.cell_max_err]
            
            ## Hi-res error
            if do_calc_hi_res_err:
                perf_0 = perf_counter()
                msg = ( '[Trial {}] Obtaining high-resolution error...\n'.format(trial)
                       )
                print_msg(msg)
                
                hr_err_ind = high_res_err(mesh, uh_proj, kappa, sigma,
                                          Phi, [u, [False, False, False]],
                                          f, solver = 'spsolve',
                                          verbose = True)
                
                perf_f = perf_counter()
                perf_diff = perf_f - perf_0
                msg = ( '[Trial {}] High-resolution error obtained!\n'.format(trial) +
                        22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                       )
                print_msg(msg)
                
                hr_errs += [hr_err_ind.max_err]
                
            ## Low-res error
            if do_calc_low_res_err:
                perf_0 = perf_counter()
                msg = ( '[Trial {}] Obtaining low-resolution error...\n'.format(trial)
                       )
                print_msg(msg)
                
                lr_err_ind = low_res_err(mesh, uh_proj, kappa, sigma,
                                         Phi, [u, [False, False, False]],
                                         f, solver = 'spsolve',
                                         verbose = True)
                
                perf_f = perf_counter()
                perf_diff = perf_f - perf_0
                msg = ( '[Trial {}] Low-resolution error obtained!\n'.format(trial) +
                        22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                       )
                print_msg(msg)
                
                lr_errs += [lr_err_ind.max_err]
                
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
                
            if do_plot_u:
                perf_0 = perf_counter()
                msg = ( '[Trial {}] Plotting analytic solution...\n'.format(trial)
                       )
                print_msg(msg)
                
                u_proj = Projection(mesh, u)
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
                
            if do_plot_diff:
                perf_0 = perf_counter()
                msg = ( '[Trial {}] Plotting difference in solutions...\n'.format(trial)
                       )
                print_msg(msg)
                
                u_vec     = u_proj.to_vector()
                uh_vec    = uh_proj.to_vector()
                diff_vec  = u_vec - uh_vec
                diff_proj = to_projection(mesh, diff_vec)
                
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
                
            if do_plot_anl_err_ind:
                perf_0 = perf_counter()
                msg = ( '[Trial {}] Plotting analytic error indicators...\n'.format(trial)
                       )
                print_msg(msg)
                
                file_name = 'anl_err_by_col_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_error_indicator(mesh, anl_err_ind,
                                     file_name = file_path,
                                     name = 'Analytic Max-Norm',
                                     by_cell = False,
                                     scale = 'pos')
                
                file_name = 'anl_err_by_cell_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_error_indicator(mesh, anl_err_ind,
                                     file_name = file_path,
                                     name = 'Analytic Max-Norm',
                                     by_col = False,
                                     scale = 'pos')
                
                anl_err_ind_ang = anl_err_ang(mesh, uh_proj, u_intg_xy)
                file_name = 'anl_err_ang_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_error_indicator(mesh, anl_err_ind_ang,
                                     file_name = file_path,
                                     name = 'Analytic Max-Norm Angular',
                                     scale = 'pos')
                
                anl_err_ind_spt = anl_err_spt(mesh, uh_proj, u_intg_th)
                file_name = 'anl_err_spt_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                plot_error_indicator(mesh, anl_err_ind_spt,
                                     file_name = file_path,
                                     name = 'Analytic Max-Norm Spatial',
                                     scale = 'pos')
                
                perf_f = perf_counter()
                perf_diff = perf_f - perf_0
                msg = ( '[Trial {}] Analytic error indicators plotted!\n'.format(trial) +
                        22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
                       )
                print_msg(msg)
                
                
            if do_plot_sol_vecs:
                # Plot solutions
                fig, ax = plt.subplots()
                
                ax.plot(u_vec_intr,
                        label = 'Analytic Solution',
                        color = 'r',
                        drawstyle = 'steps-post')
                ax.plot(uh_vec_intr,
                        label = 'Approximate Solution',
                        color = 'k', linestyle = ':',
                        drawstyle = 'steps-post')
                
                ax.legend()
                
                ax.set_title('Solution Comparison')
                
                file_name = 'soln_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                fig.set_size_inches(6.5, 6.5)
                plt.savefig(file_path, dpi = 300)
                plt.close(fig)
                
                # Plot solutions
                fig, ax = plt.subplots()
                
                max_u_vec_intr = np.amax(np.abs(u_vec_intr))
                ax.plot((u_vec_intr - uh_vec_intr) / max_u_vec_intr,
                        label = '$(u - u_{h}) / max(\abs{u})$',
                        color = 'k',
                        drawstyle = 'steps-post')
                
                ax.legend()
                
                ax.set_title('Solution Comparison')
                
                file_name = 'diff_{}.png'.format(trial)
                file_path = os.path.join(trial_dir, file_name)
                fig.set_size_inches(6.5, 6.5)
                plt.savefig(file_path, dpi = 300)
                plt.close(fig)
                
            # Refine the mesh for the next trial
            if ref_type == 'sin':
                ## Refine a given column
                col_keys = sorted(mesh.cols.keys())
                mesh.ref_col(col_keys[-1], kind = ref_kind, form = ref_form)
            elif ref_type == 'uni':
                ## Refine the mesh uniformly
                mesh.ref_mesh(kind = ref_kind, form = ref_form)
            elif ref_type == 'amr-anl':
                if ref_kind in ['ang', 'all']:
                    mesh = ref_by_ind(mesh, anl_err_ind_ang)
                if ref_kind in ['spt', 'all']:
                    mesh = ref_by_ind(mesh, anl_err_ind_spt)
            elif ref_type == 'amr-jmp':
                if ref_kind in ['ang', 'all']:
                    mesh = ref_by_ind(mesh, jmp_err_ind_ang)
                if ref_kind in ['spt', 'all']:
                    mesh = ref_by_ind(mesh, jmp_err_ind_spt)
            elif ref_type == 'rng':
                if ref_kind == 'spt':
                    kwargs = {'ref_col'      : True,
                              'col_ref_form' : ref_form,
                              'col_ref_kind' : 'spt',
                              'col_ref_tol'  : tol_spt,
                              'ref_cell'      : False}
                elif ref_kind == 'ang':
                    kwargs = {'ref_col'      : False,
                              'ref_cell'      : True,
                              'cell_ref_form' : ref_form,
                              'cell_ref_kind' : 'ang',
                              'cell_ref_tol'  : tol_ang}
                elif ref_kind == 'all':
                    kwargs = {'ref_col'      : True,
                              'col_ref_form' : ref_form,
                              'col_ref_kind' : 'spt',
                              'col_ref_tol'  : tol_spt,
                              'ref_cell'      : True,
                              'cell_ref_form' : ref_form,
                              'cell_ref_kind' : 'ang',
                              'cell_ref_tol'  : tol_ang}
                else:
                    kwargs = {}
                    
                rand_err_ind = rand_err(mesh, **kwargs)
                
                mesh = ref_by_ind(mesh, rand_err_ind)
                
            elif ref_type == 'nneg':
                if ref_kind == 'spt':
                    kwargs = {'ref_col'      : True,
                              'col_ref_form' : ref_form,
                              'col_ref_kind' : 'spt',
                              'ref_cell'      : False}
                elif ref_kind == 'ang':
                    kwargs = {'ref_col'      : False,
                              'ref_cell'      : True,
                              'cell_ref_form' : ref_form,
                              'cell_ref_kind' : 'ang'}
                elif ref_kind == 'all':
                    kwargs = {'ref_col'      : True,
                              'col_ref_form' : ref_form,
                              'col_ref_kind' : 'spt',
                              'ref_cell'      : True,
                              'cell_ref_form' : ref_form,
                              'cell_ref_kind' : 'ang'}
                else:
                    kwargs = {}
                    
                nneg_err_ind = nneg_err(mesh, uh_proj, **kwargs)
                mesh = ref_by_ind(mesh, nneg_err_ind)
                
            perf_trial_f    = perf_counter()
            perf_trial_diff = perf_trial_f - perf_trial_0
            msg = (
                '[Trial {}] Trial completed! '.format(trial) +
                'Time Elapsed: {:08.3f} [s]\n'.format(perf_trial_diff)
            )
            print_msg(msg)
            
            trial += 1
            
        if do_plot_errs:
            fig, ax = plt.subplots()
            
            colors = ['#000000', '#E69F00', '#56B4E9', '#009E73',
                      '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
                      '#882255']
            
            ax.plot(ref_ndofs, anl_errs,
                    label     = 'Analytic Error',
                    color     = colors[0],
                    linestyle = '--')
            
            ax.plot(ref_ndofs, anl_spt_errs,
                    label     = 'Analytic Spatial Error',
                    color     = colors[1],
                    linestyle = '--')
            
            ax.plot(ref_ndofs, anl_ang_errs,
                    label     = 'Analytic Angular Error',
                    color     = colors[2],
                    linestyle = '--')
            
            ax.plot(ref_ndofs, jmp_spt_errs,
                    label     = 'Column Jump Error',
                    color     = colors[3],
                    linestyle = '--')
            
            ax.plot(ref_ndofs, jmp_ang_errs,
                    label     = 'Cell Jump Error',
                    color     = colors[4],
                    linestyle = '--')
            
            if do_calc_hi_res_err:
                ax.plot(ref_ndofs, hr_errs,
                        label     = 'High-Resolution Error',
                        color     = colors[5],
                        linestyle = '-.')
                
            if do_calc_low_res_err:
                ax.plot(ref_ndofs, lr_errs,
                        label     = 'Low-Resolution Error',
                        color     = colors[6],
                        linestyle = '-.')
                
            ax.set_xscale('log', base = 2)
            ax.set_yscale('log', base = 10)
            
            errs = anl_errs + jmp_spt_errs + jmp_ang_errs + hr_errs + lr_errs
            max_err = max(errs)
            min_err = min(errs)
            if np.log2(max_err) - np.log2(min_err) < 1:
                ymin = 2**(np.floor(np.log2(min_err)))
                ymax = 2**(np.ceil(np.log2(max_err)))
                ax.set_ylim([ymin, ymax])
                
            ax.set_xlabel('Total Degrees of Freedom')
            ax.set_ylabel('Error')
            
            ax.legend()
            
            ref_strat_str = ''
            if ref_type == 'sin':
                ref_strat_str = 'Single Column'
            elif ref_type == 'uni':
                ref_strat_str = 'Uniform'
            elif ref_type == 'amr-anl':
                ref_strat_str = 'Analytic-Adaptive'
            elif ref_type == 'amr-jmp':
                ref_strat_str = 'Jump-Adaptive'
                
            ref_kind_str = ''
            if ref_kind == 'spt':
                ref_kind_str = 'Spatial'
            elif ref_kind == 'ang':
                ref_kind_str = 'Angular'
            elif ref_kind == 'all':
                ref_kind_str = 'Spatio-Angular'
                
            title_str = ( '{} {} ${}$-Refinement '.format(ref_strat_str,
                                                          ref_kind_str,
                                                          ref_form) +
                          'Convergence Rate')
            ax.set_title(title_str)
            
            file_name = '{}-convergence.png'.format(ref_form)
            file_path = os.path.join(combo_dir, file_name)
            fig.set_size_inches(6.5, 6.5)
            plt.savefig(file_path, dpi = 300)
            plt.close(fig)
            
            combo_ndofs[combo_str]    = ref_ndofs
            combo_anl_errs[combo_str] = anl_errs
            combo_hr_errs[combo_str]  = hr_errs
                
    if do_plot_errs:
        fig, ax = plt.subplots()
        
        combo_names = list(combo_ndofs.keys())
        ncombo = len(combos)
        
        colors = ['#000000', '#E69F00', '#56B4E9', '#009E73',
                  '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
                  '#882255']
        
        for cc in range(0, ncombo):
            combo_str = combo_names[cc]
            ax.plot(combo_ndofs[combo_str], combo_anl_errs[combo_str],
                    label     = combo_str + ' Analytic',
                    color     = colors[cc],
                    linestyle = '--')
            
        if do_calc_hi_res_err:
            ax.plot(combo_ndofs[combo_str], combo_hr_errs[combo_str],
                    label     = combo_str + ' High-Resolution',
                    color     = colors[cc],
                    linestyle = '-.')
            
        ax.legend()
        
        ax.set_xscale('log', base = 2)
        ax.set_yscale('log', base = 10)
        
        ax.set_xlabel('Total Degrees of Freedom')
        ax.set_ylabel('Error')
        
        title_str = ( 'Convergence Rate')
        ax.set_title(title_str)
        
        file_name = 'convergence.png'
        file_path = os.path.join(figs_dir, file_name)
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(file_path, dpi = 300)
        plt.close(fig)

if __name__ == '__main__':
    main()
