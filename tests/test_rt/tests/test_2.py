import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve, eigs
from time import perf_counter
import os, sys

from .gen_mesh import gen_mesh

sys.path.append('../../tests')
from test_cases import get_cons_prob

sys.path.append('../../src')
from dg.mesh import Mesh, get_hasnt_th
from dg.mesh.utils import plot_mesh, plot_mesh_p
from dg.matrix import get_intr_mask, split_matrix, merge_vectors
from dg.projection import Projection, push_forward, to_projection, intg_th
from dg.projection.utils import plot_xy, plot_xth, plot_yth, plot_xyth
import dg.quadrature as qd
from rt import calc_mass_matrix, calc_scat_matrix, \
    calc_intr_conv_matrix, calc_bdry_conv_matrix, \
    calc_forcing_vec
from amr import anl_err, anl_err_ang, anl_err_spt, cell_jump_err, col_jump_err, \
    rand_err, high_res_err, low_res_err, nneg_err, ref_by_ind
from amr.utils import plot_error_indicator, plot_cell_jumps

from utils import print_msg

def test_2(dir_name = 'test_rt'):
    """
    Solves constructed ("manufactured") problems, with options for sub-problems
    and different types of refinement.
    """
    
    test_dir = os.path.join(dir_name, 'test_2')
    os.makedirs(test_dir, exist_ok = True)
    
    # Test parameters:
    # Problem Name: 'mass', 'scat'tering, 'conv'ection, 'comp'lete
    prob_name = ''
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
    tol_spt = 0.90
    tol_ang = 0.90
    # Maximum number of DOFs
    max_ndof = 2**14
    # Maximum number of trials
    max_ntrial = 24
    # Which combinations of Refinement Form, Refinement Type, and Refinement Kind
    combos = [
        ['h',  'uni',     'ang'],
        ['p',  'uni',     'ang'],
        ['h',  'amr-jmp', 'ang'],
        ['hp', 'amr-jmp', 'ang']
    ]
    
    # Test Output Parameters
    do_plot_mesh        = False
    do_plot_mesh_p      = True
    do_plot_matrix      = False
    do_plot_uh          = True
    do_plot_u           = True
    do_plot_diff        = True
    do_plot_anl_err_ind = False
    do_plot_sol_vecs    = False
    do_calc_hi_res_err  = False
    do_calc_low_res_err = False
    do_plot_errs        = True
    
    prob_nums = []
    for x_num in range(2, 3):
        for y_num in range(2, 3):
            for th_num in range(3, 4):
                for scat_num in range(2, 3):
                    prob_nums += [[x_num, y_num, th_num, scat_num]]
                
    for prob_num in prob_nums:
        prob_dir = os.path.join(test_dir, str(prob_num))
        os.makedirs(prob_dir, exist_ok = True)
        
        msg = ( 'Starting problem {}...\n'.format(prob_num) )
        print_msg(msg)
        
        for prob_name in ['comp']:
            subprob_dir = os.path.join(prob_dir, prob_name)
            os.makedirs(subprob_dir, exist_ok = True)
            
            msg = ( 'Starting sub-problem {}...\n'.format(prob_name) )
            print_msg(msg)
            
            combo_ndofs = {}
            combo_anl_errs = {}
            combo_hr_errs  = {}
            
            for combo in combos:
                [ref_form, ref_type, ref_kind] = combo
                combo_str = '{}-{}-{}'.format(ref_form, ref_type, ref_kind)
                combo_dir = os.path.join(subprob_dir, combo_str)
                os.makedirs(combo_dir, exist_ok = True)
                
                msg = ( 'Starting combination {}...\n'.format(combo_str) )
                print_msg(msg)
                
                # Get the base mesh, manufactured solution
                [Lx, Ly]                   = [2., 3.]
                pbcs                       = [False, False]
                [ndof_x, ndof_y, ndof_th]  = [8, 8, 3]
                has_th                     = True
                mesh = gen_mesh(Ls     = [Lx, Ly],
                                pbcs   = pbcs,
                                ndofs  = [ndof_x, ndof_y, ndof_th],
                                has_th = has_th)

                #mesh = Mesh(Ls     = [Lx, Ly],
                #            pbcs   = pbcs,
                #            ndofs  = [ndof_x, ndof_y, ndof_th],
                #            has_th = has_th)
                #for _ in range(0, 2):
                #    mesh.ref_mesh(kind = 'ang', form = 'h')
                
                # Randomly refine to start
                for _ in range(0, 0):
                    rand_err_ind = rand_err(mesh, kind = ref_kind, form = ref_form)
                    
                    mesh = ref_by_ind(mesh, rand_err_ind,
                                      ref_ratio = tol_spt,
                                      form = ref_form)
                    
                # Perform some uniform (angular or spatial) h-refinements to start
                for _ in range(0, 0):
                    mesh.ref_mesh(kind = 'all', form = 'h')
                
                [u, kappa, sigma, Phi, f,
                 u_intg_th, u_intg_xy] = get_cons_prob(prob_name = prob_name,
                                                       prob_num  = prob_num,
                                                       mesh      = mesh)
                
                if prob_name == 'mass':
                    prob_full_name = 'Mass'
                elif prob_name == 'scat':
                    prob_full_name = 'Scattering'
                elif prob_name == 'conv':
                    prob_full_name = 'Convection'
                elif prob_name == 'comp':
                    prob_full_name = 'Complete'
                    
                # Solve the manufactured problem over several trials
                ref_ndofs = []
                anl_errs  = []
                anl_spt_errs = []
                anl_ang_errs = []
                jmp_spt_errs = []
                jmp_ang_errs = []
                hr_errs   = []
                lr_errs   = []
                
                ndof = 0
                trial = 0
                while (ndof < max_ndof) and (trial < max_ntrial):
                    perf_trial_0 = perf_counter()
                    msg = '[Trial {}] Starting with {} of {} ndofs...\n'.format(trial, ndof, max_ndof)
                    print_msg(msg)
                    
                    # Set up output directories
                    trial_dir = os.path.join(combo_dir, 'trial_{}'.format(trial))
                    os.makedirs(trial_dir, exist_ok = True)
                    
                    # Construct the matrices for the problem
                    if prob_name in ['mass', 'scat', 'comp']:
                        ## Mass matrix
                        perf_cons_0 = perf_counter()
                        msg = '[Trial {}] Constructing mass matrix...'.format(trial)
                        print_msg(msg)
                        
                        M_mass = calc_mass_matrix(mesh, kappa)
                        
                        perf_cons_f    = perf_counter()
                        perf_cons_diff = perf_cons_f - perf_cons_0
                        msg = (
                            '[Trial {}] Mass matrix constructed! '.format(trial) +
                            'Time Elapsed: {:08.3f} [s]'.format(perf_cons_diff)
                        )
                        print_msg(msg)
                    
                    if prob_name in ['scat', 'comp']:
                        ## Scattering matrix
                        perf_cons_0 = perf_counter()
                        msg = '[Trial {}] Constructing scattering matrix...'.format(trial)
                        print_msg(msg)
                        
                        M_scat = calc_scat_matrix(mesh, sigma, Phi)
                        
                        perf_cons_f    = perf_counter()
                        perf_cons_diff = perf_cons_f - perf_cons_0
                        msg = (
                            '[Trial {}] Scattering matrix constructed! '.format(trial) +
                            'Time Elapsed: {:08.3f} [s]'.format(perf_cons_diff)
                        )
                        print_msg(msg)
                        
                        
                    if prob_name in ['conv', 'comp']:
                        ## Interior convection matrix
                        perf_cons_0 = perf_counter()
                        msg = '[Trial {}] Constructing interior convection matrix...'.format(trial)
                        print_msg(msg)
                        
                        M_intr_conv = calc_intr_conv_matrix(mesh)
                        
                        perf_cons_f    = perf_counter()
                        perf_cons_diff = perf_cons_f - perf_cons_0
                        msg = (
                            '[Trial {}] Interior convection matrix constructed! '.format(trial) +
                            'Time Elapsed: {:08.3f} [s]'.format(perf_cons_diff)
                        )
                        print_msg(msg)
                        
                        ## Boundary convection matrix
                        perf_cons_0 = perf_counter()
                        msg = ('[Trial {}] Constructing '.format(trial) +
                               'boundary convection matrix...'
                               )
                        print_msg(msg)
                        
                        M_bdry_conv = calc_bdry_conv_matrix(mesh)
                        
                        perf_cons_f    = perf_counter()
                        perf_cons_diff = perf_cons_f - perf_cons_0
                        msg = (
                            '[Trial {}] Boundary convection matrix constructed! '.format(trial) +
                            'Time Elapsed: {:08.3f} [s]'.format(perf_cons_diff)
                        )
                        print_msg(msg)
                        
                    ## Forcing vector, analytic solution, interior DOFs mask
                    f_vec  = calc_forcing_vec(mesh, f)
                    u_proj = Projection(mesh, u)
                    u_vec  = u_proj.to_vector()
                    
                    intr_mask  = get_intr_mask(mesh)
                    bdry_mask  = np.invert(intr_mask)
                    f_vec_intr = f_vec[intr_mask]
                    u_vec_intr = u_vec[intr_mask]
                    bcs_vec    = u_vec[bdry_mask]
                    
                    ## Solve manufactured problem
                    perf_soln_0 = perf_counter()
                    msg = '[Trial {}] Solving manufactured problem...'.format(trial)
                    print_msg(msg)
                    
                    if prob_name == 'mass':
                        M = M_mass
                    elif prob_name == 'scat':
                        M = M_mass - M_scat
                    elif prob_name == 'conv':
                        M = M_bdry_conv - M_intr_conv
                    elif prob_name == 'comp':
                        M = (M_bdry_conv - M_intr_conv) + M_mass - M_scat
                    else:
                        msg = 'ERROR - Undefined problem name {}'.format(prob_name)
                        print(msg)
                        sys.exit(-1)
                        
                    [M_intr, M_bdry] = split_matrix(mesh, M, intr_mask)
                    
                    uh_vec_intr = spsolve(M_intr, f_vec_intr - M_bdry @ bcs_vec)
                    
                    perf_soln_f    = perf_counter()
                    perf_soln_diff = perf_soln_f - perf_soln_0
                    msg = (
                        '[Trial {}] Manufactured problem solved! '.format(trial) +
                        'Time Elapsed: {:08.3f} [s]'.format(perf_soln_diff)
                    )
                    print_msg(msg)
                    
                    uh_vec  = merge_vectors(uh_vec_intr, bcs_vec, intr_mask)
                    uh_proj = to_projection(mesh, uh_vec)
                    
                    ndof = np.size(uh_vec)
                    ref_ndofs += [ndof]
                    
                    # Caluclate error
                    ## Analytic error
                    anl_err_ind = anl_err(mesh, uh_proj, u)
                    anl_errs += [anl_err_ind.max_err]
                    
                    anl_err_ind_spt = anl_err_spt(mesh, uh_proj, u_intg_th)
                    anl_spt_errs += [anl_err_ind_spt.max_err]
                    
                    anl_err_ind_ang = anl_err_ang(mesh, uh_proj, u_intg_xy)
                    anl_ang_errs += [anl_err_ind_ang.max_err]

                    ## Jump error
                    jmp_err_ind_spt = col_jump_err(mesh, uh_proj)
                    jmp_spt_errs += [jmp_err_ind_spt.max_err]

                    jmp_err_ind_ang = cell_jump_err(mesh, uh_proj)
                    jmp_ang_errs += [jmp_err_ind_ang.max_err]
                    
                    ## Hi-res error
                    if prob_name == 'comp' and do_calc_hi_res_err:
                        hr_err_ind = high_res_err(mesh, uh_proj, kappa, sigma,
                                                  Phi, [u, [False, False, False]],
                                                  f, solver = 'spsolve',
                                                  verbose = True)
                        hr_errs += [hr_err_ind.max_err]

                    ## Low-res error
                    if prob_name == 'comp' and do_calc_low_res_err:
                        lr_err_ind = low_res_err(mesh, uh_proj, kappa, sigma,
                                                 Phi, [u, [False, False, False]],
                                                 f, solver = 'spsolve',
                                                 verbose = True)
                        lr_errs += [lr_err_ind.max_err]
                    
                    mesh_2d = get_hasnt_th(mesh)
                    anl_err_ind = anl_err(mesh, uh_proj, u)
                    
                    diff_vec_intr = uh_vec_intr - u_vec_intr
                    zero_bcs_vec  = 0. * bcs_vec
                    diff_vec      = merge_vectors(diff_vec_intr, zero_bcs_vec, intr_mask)
                    diff_proj     = to_projection(mesh, diff_vec)
                    
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
                        
                    if do_plot_matrix:
                        # Get the ending indices for the column matrices,
                        # number of DOFs in mesh
                        col_items = sorted(mesh.cols.items())
                        col_end_idxs = []
                        ncol      = 0
                        for col_key, col in col_items:
                            if col.is_lf:
                                ncol += 1
                                
                                mesh_ndof    = 0
                                for col_key, col in col_items:
                                    if col.is_lf:
                                        col_ndof         = 0
                                        [ndof_x, ndof_y] = col.ndofs
                                        
                                        cell_items = sorted(col.cells.items())
                                        for cell_key, cell in cell_items:
                                            if cell.is_lf:
                                                [ndof_th] = cell.ndofs
                                                
                                                col_ndof  += ndof_x * ndof_y * ndof_th
                                            
                                        mesh_ndof += col_ndof
                                        col_end_idxs += [mesh_ndof]
                                        
                                        
                        # Plot global matrix
                        fig, ax = plt.subplots()
                        for idx in range(0, ncol - 1):
                            ax.axhline(y         = col_end_idxs[idx],
                                       color     = 'gray',
                                       linestyle = '--',
                                       linewidth = 0.2)
                            ax.axvline(x         = col_end_idxs[idx],
                                       color     = 'gray',
                                       linestyle = '--',
                                       linewidth = 0.2)
                        ax.spy(M,
                               marker     = 's',
                               markersize = 0.2,
                               color      = 'k')
                        
                        title_str = 'Global {} Matrix'.format(prob_full_name)
                        ax.set_title(title_str)
                        
                        file_name = '{}_matrix_{}.png'.format(prob_name, trial)
                        file_path = os.path.join(trial_dir, file_name)
                        fig.set_size_inches(6.5, 6.5)
                        plt.savefig(file_path, dpi = 300)
                        plt.close(fig)
                        
                        # Plot eigenvalues
                        ndof = int(M.get_shape()[0])
                        [evals, _] = eigs(M, k = ndof - 2, which = 'LM')
                        evals = sorted(np.abs(evals))
                        fig, ax = plt.subplots()
                        ax.plot(evals, 'b.')
                        file_name = '{}_evals_{}.png'.format(prob_name, trial)
                        file_path = os.path.join(trial_dir, file_name)
                        fig.set_size_inches(6.5, 6.5)
                        plt.savefig(file_path, dpi = 300)
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

                        file_name = 'uh_xyth_{}.png'.format(trial)
                        file_path = os.path.join(trial_dir, file_name)
                        plot_xyth(mesh, uh_proj, file_name = file_path)
                        
                    if do_plot_u:
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
                        
                    if do_plot_diff:            
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
                    
                    if do_plot_anl_err_ind:
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
                            mesh = ref_by_ind(mesh, anl_err_ind_ang,
                                              ref_ratio = tol_ang, form = ref_form)
                        if ref_kind in ['spt', 'all']:
                            mesh = ref_by_ind(mesh, anl_err_ind_spt,
                                              ref_ratio = tol_spt, form = ref_form)
                    elif ref_type == 'amr-jmp':
                        if ref_kind in ['ang', 'all']:
                            mesh = ref_by_ind(mesh, jmp_err_ind_ang,
                                              ref_ratio = tol_ang, form = ref_form)
                        if ref_kind in ['spt', 'all']:
                            mesh = ref_by_ind(mesh, jmp_err_ind_spt,
                                              ref_ratio = tol_spt, form = ref_form)
                    elif ref_type == 'rng':
                        rand_err_ind = rand_err(mesh,
                                                kind = ref_kind,
                                                form = ref_form)
                        
                        mesh = ref_by_ind(mesh, rand_err_ind,
                                          ref_ratio = tol_spt,
                                          form = ref_form)
                        
                    elif ref_type == 'rng2':
                        col_keys = list(mesh.cols.keys())
                        ncol = len(col_keys)
                        to_ref = np.random.choice(col_keys,
                                                  max(1, int(ncol * (1. - tol_spt))),
                                                  replace = False)
                        for col_key in to_ref:
                            mesh.ref_col(col_key, kind = 'ang', form = ref_form)
                    elif ref_type == 'nneg':
                        nneg_err_ind = nneg_err(mesh, uh_proj,
                                                kind = ref_kind,
                                                form = ref_form)
                        mesh = ref_by_ind(mesh, nneg_err_ind,
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
                    
                    if prob_name == 'comp' and do_calc_hi_res_err:
                        ax.plot(ref_ndofs, hr_errs,
                                label     = 'High-Resolution Error',
                                color     = colors[5],
                                linestyle = '-.')

                    if prob_name == 'comp' and do_calc_low_res_err:
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
                    ax.set_ylabel('L$^{\infty}$ Error')
                    
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
                                  'Convergence Rate - {} Problem'.format(prob_full_name) )
                    ax.set_title(title_str)
                    
                    file_name = '{}-convergence-{}.png'.format(ref_form,
                                                               prob_full_name.lower())
                    file_path = os.path.join(combo_dir, file_name)
                    fig.set_size_inches(6.5, 6.5)
                    plt.savefig(file_path, dpi = 300)
                    plt.close(fig)
                    
                combo_ndofs[combo_str] = ref_ndofs
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
                    
                    if prob_name == 'comp' and do_calc_hi_res_err:
                        ax.plot(combo_ndofs[combo_str], combo_hr_errs[combo_str],
                                label     = combo_str + ' High-Resolution',
                                color     = colors[cc],
                                linestyle = '-.')
                    
                ax.legend()
                
                ax.set_xscale('log', base = 2)
                ax.set_yscale('log', base = 10)
                
                ax.set_xlabel('Total Degrees of Freedom')
                ax.set_ylabel('L$^{\infty}$ Error')
                
                title_str = ( 'Convergence Rate - {} Problem'.format(prob_full_name) )
                ax.set_title(title_str)
                
                file_name = 'convergence-{}_{}.png'.format(prob_full_name.lower(),
                                                           prob_num)
                file_path = os.path.join(subprob_dir, file_name)
                fig.set_size_inches(6.5, 6.5)
                plt.savefig(file_path, dpi = 300)
                plt.close(fig)
