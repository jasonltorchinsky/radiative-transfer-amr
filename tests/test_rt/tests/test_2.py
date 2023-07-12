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
from amr import total_anl_err, anl_err_ang, anl_err_spt, cell_jump_err, col_jump_err, \
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
     # AMR Refinement Tolerance
    tol_spt = 0.90
    tol_ang = 0.90
    # Maximum number of DOFs
    max_ndof = 2**16
    # Maximum number of trials
    max_ntrial = 64
    # Which combinations of Refinement Form, Refinement Type, and Refinement Kind
    combos = [
        ['h', 'uni', 'ang'],
        ['p', 'uni', 'ang']
    ]
    
    # Test Output Parameters
    do_plot_mesh_p      = True
    do_plot_uh          = True
    do_plot_u           = True
    do_plot_diff        = True
    do_plot_anl_err_ind = False
    do_plot_sol_vecs    = False
    do_calc_hi_res_err  = False
    do_calc_low_res_err = False
    do_plot_errs        = True

    # Which problems to solve
    prob_nums = []
    for x_num in range(2, 3):
        for y_num in range(2, 3):
            for th_num in range(0, 4):
                for scat_num in range(0, 3):
                    prob_nums += [[x_num, y_num, th_num, scat_num]]
    # Which sub-problems to solve
    sub_probs = ['comp']
                    
    for prob_num in [[2, 2, 3, 0], [2, 2, 3, 1], [2, 2, 3, 2]]:
        prob_dir = os.path.join(test_dir, str(prob_num))
        os.makedirs(prob_dir, exist_ok = True)
        
        msg = ( 'Starting problem {}...\n'.format(prob_num) )
        print_msg(msg)
        
        for prob_name in sub_probs:
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
                [ndof_x, ndof_y, ndof_th]  = [8, 8, 2]
                has_th                     = True
                mesh = gen_mesh(Ls     = [Lx, Ly],
                                pbcs   = pbcs,
                                ndofs  = [ndof_x, ndof_y, ndof_th],
                                has_th = has_th)
                
                [u, kappa, sigma, Phi, f,
                 u_intg_th, u_intg_xy] = get_cons_prob(prob_name = prob_name,
                                                       prob_num  = prob_num,
                                                       mesh      = mesh,
                                                       sth       = 64.)
                
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
                
                ndof  = mesh.get_ndof()
                trial = 0
                while (ndof < max_ndof) and (trial < max_ntrial):
                    ndof = mesh.get_ndof()
                    ref_ndofs += [ndof]
                    
                    perf_trial_0 = perf_counter()
                    msg = '[Trial {}] Starting with {} of {} ndofs...\n'.format(trial, ndof, max_ndof)
                    print_msg(msg)
                    
                    # Set up output directories
                    trial_dir = os.path.join(combo_dir, 'trial_{}'.format(trial))
                    os.makedirs(trial_dir, exist_ok = True)
                    
                    # Construct the matrices for the problem
                    if prob_name in ['mass', 'scat', 'comp']:
                        ## Mass matrix
                        M_mass = get_mass_matrix(mesh, kappa, trial)
                    
                    if prob_name in ['scat', 'comp']:
                        ## Scattering matrix
                        M_scat = get_scat_matrix(mesh, sigma, Phi, trial)
                        
                    if prob_name in ['conv', 'comp']:
                        ## Convection matrix
                        M_conv = get_conv_matrix(mesh, trial)
                        
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
                        M = M_conv
                    elif prob_name == 'comp':
                        M = M_conv + M_mass - M_scat
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
                        
                    #anl_err_ind = anl_err(mesh, uh_proj, u, **kwargs)
                    anl_err = total_anl_err(mesh, uh_proj, u)

                    anl_errs += [anl_err]
                    #if ref_kind == 'spt':
                    #    anl_errs += [anl_err_ind.col_max_err]
                    #elif ref_kind == 'ang':
                    #    anl_errs += [anl_err_ind.cell_max_err]
                    #elif ref_kind == 'all':
                    #    anl_errs += [max(anl_err_ind.col_max_err, anl_err_ind.cell_max_err)]
                    #else:
                    #    anl_errs += [anl_err_ind.max_col_err]
                        
                    ## Analytic angularly-integrated error
                    #kwargs = {'ref_col'      : True,
                    #          'col_ref_form' : ref_form,
                    #          'col_ref_kind' : 'spt',
                    #          'col_ref_tol'  : tol_spt,
                    #          'ref_cell'      : False}
                    #anl_err_ind_spt = anl_err_spt(mesh, uh_proj, u_intg_th, **kwargs)
                    #anl_spt_errs += [anl_err_ind_spt.col_max_err]
                    
                    ## Analytic spatially-integrated error
                    #kwargs = {'ref_col'      : False,
                    #          'ref_cell'      : True,
                    #          'cell_ref_form' : ref_form,
                    #          'cell_ref_kind' : 'ang',
                    #          'cell_ref_tol'  : tol_ang}
                    #anl_err_ind_ang = anl_err_ang(mesh, uh_proj, u_intg_xy, **kwargs)
                    #anl_ang_errs += [anl_err_ind_ang.cell_max_err]
                    
                    ## Jump error
                    #kwargs = {'ref_col'      : True,
                    #          'col_ref_form' : ref_form,
                    #          'col_ref_kind' : 'spt',
                    #          'col_ref_tol'  : tol_spt,
                    #          'ref_cell'      : False}
                    #jmp_err_ind_spt = col_jump_err(mesh, uh_proj, **kwargs)
                    #jmp_spt_errs += [jmp_err_ind_spt.col_max_err]
                    
                    #kwargs = {'ref_col'      : False,
                    #          'ref_cell'      : True,
                    #          'cell_ref_form' : ref_form,
                    #          'cell_ref_kind' : 'ang',
                    #          'cell_ref_tol'  : tol_ang}
                    #jmp_err_ind_ang = cell_jump_err(mesh, uh_proj, **kwargs)
                    #jmp_ang_errs += [jmp_err_ind_ang.cell_max_err]
                    
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
                        diff_vec_intr = uh_vec_intr - u_vec_intr
                        zero_bcs_vec  = 0. * bcs_vec
                        diff_vec      = merge_vectors(diff_vec_intr, zero_bcs_vec, intr_mask)
                        diff_proj     = to_projection(mesh, diff_vec)
                    
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
                    
                    ax.scatter(ref_ndofs, anl_errs,
                               label     = None,
                               color     = colors[0])

                    # Get best-fit line
                    [a, b] = np.polyfit(np.log10(ref_ndofs), np.log10(anl_errs), 1)
                    xx = np.logspace(np.log10(ref_ndofs[0]), np.log10(ref_ndofs[-1]))
                    yy = 10**b * xx**a
                    ax.plot(xx, yy,
                            label = '{} Analytic : {:4.2f}'.format(combo_str, a),
                            color = colors[0],
                            linestyle = '--'
                            )
                    
                    #ax.plot(ref_ndofs, anl_spt_errs,
                    #        label     = 'Analytic Spatial Error',
                    #        color     = colors[1],
                    #        linestyle = '--')
                    
                    #ax.plot(ref_ndofs, anl_ang_errs,
                    #        label     = 'Analytic Angular Error',
                    #        color     = colors[2],
                    #        linestyle = '--')
                    
                    #ax.plot(ref_ndofs, jmp_spt_errs,
                    #        label     = 'Column Jump Error',
                    #        color     = colors[3],
                    #        linestyle = '--')
                    
                    #ax.plot(ref_ndofs, jmp_ang_errs,
                    #        label     = 'Cell Jump Error',
                    #        color     = colors[4],
                    #        linestyle = '--')
                    
                    #if prob_name == 'comp' and do_calc_hi_res_err:
                    #    ax.plot(ref_ndofs, hr_errs,
                    #            label     = 'High-Resolution Error',
                    #            color     = colors[5],
                    #            linestyle = '-.')

                    #if prob_name == 'comp' and do_calc_low_res_err:
                    #    ax.plot(ref_ndofs, lr_errs,
                    #            label     = 'Low-Resolution Error',
                    #            color     = colors[6],
                    #            linestyle = '-.')
                    
                    ax.set_xscale('log', base = 10)
                    ax.set_yscale('log', base = 10)
                    
                    errs = anl_errs #+ jmp_spt_errs + jmp_ang_errs #+ hr_errs + lr_errs
                    max_err = max(errs)
                    min_err = min(errs)
                    if np.log10(max_err) - np.log10(min_err) < 1:
                        ymin = 10**(np.floor(np.log10(min_err)))
                        ymax = 10**(np.ceil(np.log10(max_err)))
                        ax.set_ylim([ymin, ymax])
                        
                    ax.set_xlabel('Total Degrees of Freedom')
                    ax.set_ylabel(r'$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u - u_{hp} \right)^2\,d\vec{x}\,d\hat{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\hat{s}}}$')
                    
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
                    plt.tight_layout()
                    plt.savefig(file_path, dpi = 300)
                    plt.close(fig)
                    
                combo_ndofs[combo_str] = ref_ndofs
                combo_anl_errs[combo_str] = anl_errs
                #combo_hr_errs[combo_str]  = hr_errs
                
            if do_plot_errs:
                fig, ax = plt.subplots()
                
                combo_names = list(combo_ndofs.keys())
                ncombo = len(combos)
                
                colors = ['#000000', '#E69F00', '#56B4E9', '#009E73',
                          '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
                          '#882255']
                
                for cc in range(0, ncombo):
                    combo_str = combo_names[cc]
                    ndofs = combo_ndofs[combo_str]
                    errs = combo_anl_errs[combo_str]
                    
                    ax.scatter(ndofs, errs,
                               label     = None,#combo_str + ' Analytic',
                               color     = colors[cc])

                    # Get best-fit line
                    [a, b] = np.polyfit(np.log10(ndofs), np.log10(errs), 1)
                    xx = np.logspace(np.log10(ndofs[0]), np.log10(ndofs[-1]))
                    yy = 10**b * xx**a
                    ax.plot(xx, yy,
                            label = '{} Analytic : {:4.2f}'.format(combo_str, a),
                            color = colors[cc],
                            linestyle = '--'
                            )
                    
                    
                    if prob_name == 'comp' and do_calc_hi_res_err:
                        ax.plot(combo_ndofs[combo_str], combo_hr_errs[combo_str],
                                label     = combo_str + ' High-Resolution',
                                color     = colors[cc],
                                linestyle = '-.')
                    
                ax.legend()
                
                ax.set_xscale('log', base = 10)
                ax.set_yscale('log', base = 10)
                
                ax.set_xlabel('Total Degrees of Freedom')
                ax.set_ylabel(r'$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u - u_{hp} \right)^2\,d\vec{x}\,d\hat{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\hat{s}}}$')
                
                title_str = ( 'Convergence Rate - {} Problem'.format(prob_full_name) )
                ax.set_title(title_str)
                
                file_name = 'convergence-{}_{}.png'.format(prob_full_name.lower(),
                                                           prob_num)
                file_path = os.path.join(subprob_dir, file_name)
                fig.set_size_inches(6.5, 6.5)
                plt.tight_layout()
                plt.savefig(file_path, dpi = 300)
                plt.close(fig)

def get_mass_matrix(mesh, kappa, trial):
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
    
    return M_mass

def get_scat_matrix(mesh, sigma, Phi, trial):
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

    return M_scat

def get_conv_matrix(mesh, trial):
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

    return M_bdry_conv - M_intr_conv
