import copy
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import os, sys

from .gen_mesh import gen_mesh

sys.path.append('../../tests')
from test_cases import get_test_prob

sys.path.append('../../src')
from dg.mesh import get_hasnt_th
from dg.mesh.utils import plot_mesh, plot_mesh_p
from dg.projection import Projection, intg_th
from dg.projection.utils import plot_projection, plot_xy, plot_xth, plot_yth
from rt import rtdg
from amr import cell_jump_err, col_jump_err, rand_err, high_res_err, ref_by_ind

from utils import print_msg


def test_3(dir_name = 'test_rt'):
    """
    Solves test problems, using the built-in RT solver.
    """
    
    test_dir = os.path.join(dir_name, 'test_3')
    os.makedirs(test_dir, exist_ok = True)

    # Test parameters:
    # Problem Number
    prob_num  = None
    # Refinement Type: 'sin'gle column, 'uni'form, 'a'daptive 'm'esh 'r'efinement,
    # random ('rng')
    ref_type = ''
    # Refinement Kind: 's'pa't'ia'l', 'ang'ular, 'all'
    ref_kind = ''
    # Refinement Form: 'h', 'p'
    ref_form = ''
    # AMR Refinement Tolerance
    tol_spt = 0.85
    tol_ang = 0.85
    # Maximum number of DOFs
    max_ndof = 2**15
    # Maximum number of trials
    max_ntrial = 4
    # Which combinations of Refinement Form, Refinement Type, and Refinement Kind
    combos = [
        ['h',  'rng', 'spt'],
        ['p',  'rng', 'spt'],
        ['hp', 'rng', 'spt']
    ]

    # Test Output Parameters
    do_plot_mesh        = False
    do_plot_mesh_p      = True
    do_plot_coeff_funcs = False
    do_plot_uh          = True
    do_plot_errs        = True

    prob_nums = [0, 1]

    for prob_num in prob_nums:
        prob_dir = os.path.join(test_dir, str(prob_num))
        os.makedirs(prob_dir, exist_ok = True)

        msg = ( 'Starting problem {}...\n'.format(prob_num) )
        print_msg(msg)

        combo_ndofs = {}
        combo_errs = {}

        for combo in combos:
            [ref_form, ref_type, ref_kind] = combo
            combo_str = '{}-{}-{}'.format(ref_form, ref_type, ref_kind)
            combo_dir = os.path.join(prob_dir, combo_str)
            os.makedirs(combo_dir, exist_ok = True)
            
            msg = ( 'Starting combination {}...\n'.format(combo_str) )
            print_msg(msg)
            
            # Get the base mesh, test problem
            [Lx, Ly]                   = [2., 3.]
            pbcs                       = [True, False]
            [ndof_x, ndof_y, ndof_th]  = [3, 3, 3]
            has_th                     = True
            mesh = gen_mesh(Ls     = [Lx, Ly],
                            pbcs   = pbcs,
                            ndofs  = [ndof_x, ndof_y, ndof_th],
                            has_th = has_th)
            
            # Randomly refine to start
            for _ in range(0, 0):
                rand_err_ind = rand_err(mesh, kind = ref_kind, form = ref_form)
                
                mesh = ref_by_ind(mesh, rand_err_ind,
                                  ref_ratio = tol_spt,
                                  form = ref_form)
            
            [kappa, sigma, Phi, [bcs, dirac], f] = get_test_prob(
                prob_num = prob_num,
                mesh     = mesh)
            
            # Solve the manufactured problem over several trials
            ref_ndofs = []
            inf_errs  = []
            
            ndof = 0
            trial = 0
            while (ndof < max_ndof) and (trial <= max_ntrial):
                perf_trial_0 = perf_counter()
                print_msg('[Trial {}] Starting...'.format(trial))
                
                # Set up output directories
                trial_dir = os.path.join(combo_dir, 'trial_{}'.format(trial))
                os.makedirs(trial_dir, exist_ok = True)
                
                # Solve the test problem
                perf_cons_0 = perf_counter()
                print_msg('[Trial {}] Solving the test problem...'.format(trial))
                
                uh_proj = rtdg(mesh, kappa, sigma, Phi, [bcs, dirac], f, 
                               solver = 'spsolve', verbose = True)
                
                perf_cons_f    = perf_counter()
                perf_cons_diff = perf_cons_f - perf_cons_0
                msg = (
                    '[Trial {}] Test problem solved! '.format(trial) +
                    'Time Elapsed: {:08.3f} [s]'.format(perf_cons_diff)
                )
                print_msg(msg)
                
                uh_vec = uh_proj.to_vector()
                ndof = np.size(uh_vec)
                ref_ndofs += [ndof]
                
                msg = (
                    '[Trial {}] Number of DOFs: {} '.format(trial, ndof) +
                    'of {}.'.format(max_ndof)
                )
                print_msg(msg)
                
                mesh_2d = get_hasnt_th(mesh)

                # Calculate error
                hr_err_ind = high_res_err(mesh, uh_proj, kappa, sigma,
                                          Phi, [bcs, dirac], f, solver = 'spsolve',
                                          verbose = True)
                inf_errs += [hr_err_ind.max_err]
                
                if do_plot_mesh:
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
                    
                if do_plot_mesh_p:
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
                    
                if do_plot_coeff_funcs:
                    # Plot the coefficient functions
                    kappa_proj = Projection(mesh_2d, kappa)
                    file_name = os.path.join(trial_dir, 'kappa.png')
                    plot_projection(mesh_2d, kappa_proj, file_name = file_name)
                    
                    sigma_proj = Projection(mesh_2d, sigma)
                    file_name = os.path.join(trial_dir, 'sigma.png')
                    plot_projection(mesh_2d, sigma_proj, file_name = file_name)
                    
                    file_name = os.path.join(trial_dir, 'scat.png')
                    fig, ax = plt.subplots(subplot_kw = {'projection' : 'polar'})
                    phi = np.linspace(0, 2 * np.pi, 120)
                    r = Phi(0, phi)
                    ax.plot(phi, r, 'k-')
                    ax.set_yticks(np.around(np.linspace(0, np.amax(r), 3), 2))
                    fig.set_size_inches(6.5, 6.5)
                    plt.savefig(file_name, dpi = 300)
                    plt.close(fig)
                
                if do_plot_uh:
                    file_name = 'uh_xy_{}.png'.format(trial)
                    file_path = os.path.join(trial_dir, file_name)
                    plot_xy(mesh, uh_proj, file_name = file_path)
                    
                    file_name = 'uh_xth_{}.png'.format(trial)
                    file_path = os.path.join(trial_dir, file_name)
                    plot_xth(mesh, uh_proj, file_name = file_path)
                    
                    file_name = 'uh_yth_{}.png'.format(trial)
                    file_path = os.path.join(trial_dir, file_name)
                    plot_yth(mesh, uh_proj, file_name = file_path)
                    
                    
                # Refine the mesh for the next trial
                if ref_type == 'sin':
                    ## Refine a given column
                    col_keys = sorted(mesh.cols.keys())
                    mesh.ref_col(col_keys[-1], kind = ref_kind, form = ref_form)
                elif ref_type == 'uni':
                    ## Refine the mesh uniformly
                    mesh.ref_mesh(kind = ref_kind, form = ref_form)
                elif ref_type == 'amr':
                    if ref_kind in ['ang', 'all']:
                        cell_jump_err_ind = cell_jump_err(mesh, uh_proj)
                    if ref_kind in ['spt', 'all']:
                        col_jump_err_ind = col_jump_err(mesh, uh_proj)
                    
                    if ref_kind in ['ang', 'all']:
                        mesh = ref_by_ind(mesh, cell_jump_err_ind,
                                          ref_ratio = tol, form = ref_form)
                    if ref_kind in ['spt', 'all']:
                        mesh = ref_by_ind(mesh, col_jump_err_ind,
                                          ref_ratio = tol, form = ref_form)
                elif ref_type == 'rng':
                        rand_err_ind = rand_err(mesh, kind = ref_kind,
                                                form = ref_form)
                        
                        mesh = ref_by_ind(mesh, rand_err_ind,
                                          ref_ratio = tol_spt,
                                          form = ref_form)
                        
                perf_trial_f    = perf_counter()
                perf_trial_diff = perf_trial_f - perf_trial_0
                msg = (
                    '[Trial {}] Trial completed! '.format(trial) +
                    'Time Elapsed: {:08.3f} [s]\n'.format(perf_trial_diff)
                )
                print_msg(msg)
                
                trial += 1
                
            combo_ndofs[combo_str] = ref_ndofs
            combo_errs[combo_str] = inf_errs

        if do_plot_errs:
            fig, ax = plt.subplots()
            
            combo_names = list(combo_ndofs.keys())
            ncombo = len(combos)
            
            colors = ['#000000', '#E69F00', '#56B4E9', '#009E73',
                      '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
                      '#882255']
            
            for cc in range(0, ncombo):
                combo_str = combo_names[cc]
                ax.plot(combo_ndofs[combo_str], combo_errs[combo_str],
                        label     = combo_str,
                        color     = colors[cc],
                        linestyle = '-')
            ax.legend()
            
            ax.set_xscale('log', base = 2)
            ax.set_yscale('log', base = 2)
            
            ax.set_xlabel('Total Degrees of Freedom')
            ax.set_ylabel('L$^{\infty}$ Error')
            
            title_str = ( 'Convergence Rate' )
            ax.set_title(title_str)
            
            file_name = 'convergence-{}.png'.format(prob_num)
            file_path = os.path.join(prob_dir, file_name)
            fig.set_size_inches(6.5, 6.5)
            plt.savefig(file_path, dpi = 300)
            plt.close(fig)
