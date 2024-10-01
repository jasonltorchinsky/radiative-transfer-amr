import os, sys
src_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, os.pardir, "src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Standard Library Imports
import json

from time import perf_counter

# Third-Party Library Imports
import numpy as np

# Local Library Imports
import amr
import amr.utils
import utils

import params

from gen_err_ind_plot import gen_err_ind_plot

def ref_mesh(ref_strat, mesh, uh_proj, comm_rank, rng, do_plot_err_ind, trial,
             do_calc_err, trial_dir, ndof, info, ref_strat_dir, refs, u_intg_xy):    
    # Refine the mesh, plot error indicators along the way
    if   ref_strat['short_name'] == 'h-uni-ang':
        if comm_rank == 0:
            mesh.ref_mesh(kind = 'ang', form = 'h')
        err_ind = None
    elif ref_strat['short_name'] == 'p-uni-ang':
        if comm_rank == 0:
            for _ in range(0, 3):
                mesh.ref_mesh(kind = 'ang', form = 'p')
        err_ind = None
    elif ref_strat['short_name'] == 'hp-uni-ang':
        if comm_rank == 0:
            for _ in range(0, 2):
                mesh.ref_mesh(kind = 'ang', form = 'p')
            mesh.ref_mesh(kind = 'ang', form = 'h')
        err_ind = None
    elif ref_strat['short_name'] == 'h-uni-spt':
        if comm_rank == 0:
            mesh.ref_mesh(kind = 'spt', form = 'h')
        err_ind = None
    elif ref_strat['short_name'] == 'p-uni-spt':
        if comm_rank == 0:
            for _ in range(0, 3):
                mesh.ref_mesh(kind = 'spt', form = 'p')
        err_ind = None
    elif ref_strat['short_name'] == 'hp-uni-spt':
        if comm_rank == 0:
            for _ in range(0, 2):
                mesh.ref_mesh(kind = 'spt', form = 'p')
            mesh.ref_mesh(kind = 'spt', form = 'h')
        err_ind = None
    elif ref_strat['short_name'] == 'h-uni-all':
        if comm_rank == 0:
            mesh.ref_mesh(kind = 'all', form = 'h')
        err_ind = None
    elif ref_strat['short_name'] == 'p-uni-all':
        if comm_rank == 0:
            for _ in range(0, 3):
                mesh.ref_mesh(kind = 'all', form = 'p')
        err_ind = None
    elif ref_strat['short_name'] == 'hp-uni-all':
        if comm_rank == 0:
            for _ in range(0, 2):
                mesh.ref_mesh(kind = 'all', form = 'p')
            mesh.ref_mesh(kind = 'all', form = 'h')
        err_ind = None
    elif ((ref_strat['short_name'] == 'h-amr-ang')
          or (ref_strat['short_name'] == 'p-amr-ang')
          or (ref_strat['short_name'] == 'hp-amr-ang')):
        if comm_rank == 0:
            # Have two choices: Refine based on nneg or jump, and if
            # also refining in space
            if ref_strat['ref_kind'] == 'all':
                # Calculate nneg errors
                kwargs_ang_nneg  = ref_strat['kwargs_ang_nneg']
                cell_ref_tol     = kwargs_ang_nneg['cell_ref_tol']
                nneg_ang_err_ind = amr.nneg_err_ang(mesh, uh_proj,
                                                            **kwargs_ang_nneg)
                        
                # Spatial refinement strategy is uniform p-refinement
                col_ref_tol = 1. # Positive refinement tolerance for
                # nneg error means we'll refine everything
                kwargs_spt_nneg = {'ref_col'      : True,
                                   'col_ref_form' : 'p',
                                   'col_ref_kind' : 'spt',
                                   'col_ref_tol'  : col_ref_tol,
                                   'ref_cell'      : False,
                                   'cell_ref_form' : None,
                                   'cell_ref_kind' : None,
                                   'cell_ref_tol'  : None}
                nneg_spt_err_ind = amr.nneg_err_spt(mesh, uh_proj,
                                                    **kwargs_spt_nneg)
                        
                # If nneg error exceeds tolerance, we refine whichever
                # has larger negative values in columns to be refined.
                if ((nneg_ang_err_ind.cell_max_err < cell_ref_tol)
                    or (nneg_spt_err_ind.col_max_err < cell_ref_tol)):
                    # Use cell_ref_tol here because otherwise this
                    # would always trigger.
                    err_ind_ang = nneg_ang_err_ind
                    err_ind_spt = nneg_spt_err_ind

                    avg_cell_ref_err = nneg_ang_err_ind.avg_cell_ref_err
                    avg_col_ref_err  = nneg_spt_err_ind.avg_col_ref_err
                    if np.abs(avg_cell_ref_err) > np.abs(avg_col_ref_err):
                        err_ind = nneg_ang_err_ind
                        ref_str = 'ang_nneg'
                    else:
                        err_ind = nneg_spt_err_ind
                        ref_str = 'spt_nneg'
                                
                else:
                    # If not using nneg err, then we use the jump error
                    kwargs_ang_jmp  = ref_strat['kwargs_ang_jmp']
                    jmp_ang_err_ind = amr.cell_jump_err(mesh, uh_proj,
                                                        **kwargs_ang_jmp)
                    col_ref_tol = -1. # Want uniform refinement
                    kwargs_spt_jmp = {'ref_col'      : True,
                                      'col_ref_form' : 'p',
                                      'col_ref_kind' : 'spt',
                                      'col_ref_tol'  : col_ref_tol,
                                      'ref_cell'      : False,
                                      'cell_ref_form' : None,
                                      'cell_ref_kind' : None,
                                      'cell_ref_tol'  : None}
                    jmp_spt_err_ind = amr.col_jump_err(mesh, uh_proj,
                                                       **kwargs_spt_jmp)
                    msg = (
                        'Comparison of average refinement error:\n' +
                        12 * ' ' + 'Angular {:.4E} '.format(jmp_ang_err_ind.avg_cell_ref_err) +
                        'vs. Spatial {:.4E}\n'.format(jmp_spt_err_ind.avg_col_ref_err)
                    )
                    utils.print_msg(msg, blocking = False)
                    err_ind_ang = jmp_ang_err_ind
                    err_ind_spt = jmp_spt_err_ind
                            
                    avg_cell_ref_err = jmp_ang_err_ind.avg_cell_ref_err
                    avg_col_ref_err  = jmp_spt_err_ind.avg_col_ref_err
                            
                    # Randomly choose which, but probability depends on steering criterion
                    p_ang = avg_cell_ref_err / (avg_cell_ref_err + avg_col_ref_err)
                    ref_strs = ['ang_jmp', 'spt_jmp']
                    ref_str = rng.choice(ref_strs, size = 1, p = (p_ang, 1 - p_ang))[0]
                    #if p_ang >= 0.5:
                    #    ref_str = ref_strs[0]
                    #else:
                    #    ref_str = ref_strs[1]
                    if ref_str == 'ang_jmp':
                        err_ind = jmp_ang_err_ind
                    else: # ref_str == 'spt_jmp'
                        err_ind = jmp_spt_err_ind

                if do_plot_err_ind and (trial%10 == 0 or do_calc_err):
                    gen_err_ind_plot(mesh, err_ind_ang, trial, trial_dir, 'err_ind_ang.png')
                    gen_err_ind_plot(mesh, err_ind_spt, trial, trial_dir, 'err_ind_spt.png')
                avg_cell_ref_err_str = '{:.4E}'.format(jmp_ang_err_ind.avg_cell_ref_err)
                avg_col_ref_err_str = '{:.4E}'.format(jmp_spt_err_ind.avg_col_ref_err)
                refs += [[ndof, ref_str, avg_cell_ref_err_str, avg_col_ref_err_str, info]]
            else: # Just refining in angle
                # Calculate nneg error
                kwargs_ang_nneg  = ref_strat['kwargs_ang_nneg']
                cell_ref_tol     = kwargs_ang_nneg['cell_ref_tol']
                nneg_ang_err_ind = amr.nneg_err_ang(mesh, uh_proj,
                                                    **kwargs_ang_nneg)
                        
                # If nneg error exceeds tolerance, we refine whichever
                # has larger negative values in columns to be refined.
                if (nneg_ang_err_ind.cell_max_err < cell_ref_tol):
                    err_ind = nneg_ang_err_ind
                    ref_str = 'ang_nneg'
                            
                else:
                    # If not using nneg err, then we use the jump error
                    kwargs_ang_jmp  = ref_strat['kwargs_ang_jmp']
                    jmp_ang_err_ind = amr.cell_jump_err(mesh, uh_proj,
                                                        **kwargs_ang_jmp)
                            
                    err_ind = jmp_ang_err_ind
                    ref_str = 'ang_jmp'
                if do_plot_err_ind and (trial%5 == 0 or do_calc_err):
                    gen_err_ind_plot(mesh, err_ind, trial, trial_dir, 'err_ind_ang.png')
                    # Also plot the analytic error indicator, which is the max-norm
                    # difference across each cell
                    anl_err_ind = amr.anl_err_ang(mesh, uh_proj, u_intg_xy, **kwargs_ang_jmp)
                    gen_err_ind_plot(mesh, anl_err_ind, trial, trial_dir, 'anl_err_ang.png')
                    # Also plot high resolution error indicator
                    ## CONTINUE FROM HERE
                    hr_err_ind = amr.high_res_err(mesh, uh_proj, 
                                                  params.kappa, params.sigma,
                                                  params.Phi, params.bcs_dirac, params.f,
                                                  verbose = False, dir_name = trial_dir)
                refs += [[ndof, ref_str, info]]
                    
            file_name = 'refs.txt'
            file_path = os.path.join(ref_strat_dir, file_name)
            json.dump(refs, open(file_path, 'w'))

            msg = (
                '[Trial {}] Refining cause: {}\n'.format(trial, ref_str)
                )
            utils.print_msg(msg, blocking = False)
            mesh = amr.ref_by_ind(mesh, err_ind)
                    
    elif ((ref_strat['short_name'] == 'h-amr-spt')
          or (ref_strat['short_name'] == 'p-amr-spt')
          or (ref_strat['short_name'] == 'hp-amr-spt')):
        if comm_rank == 0:
            # Have two choices: Refine based on nneg or jump, and if
            # also refining in space
            if ref_strat['ref_kind'] == 'all':
                # Calculate nneg errors
                kwargs_spt_nneg  = ref_strat['kwargs_spt_nneg']
                col_ref_tol      = kwargs_spt_nneg['col_ref_tol']
                nneg_spt_err_ind = amr.nneg_err_spt(mesh, uh_proj,
                                                    **kwargs_spt_nneg)
                        
                # Angular refinement strategy is uniform p-refinement
                cell_ref_tol = 1. # Positive refinement tolerance for
                # nneg error means we'll refine everything
                kwargs_ang_nneg = {'ref_col'      : False,
                                   'col_ref_form' : None,
                                   'col_ref_kind' : None,
                                   'col_ref_tol'  : None,
                                   'ref_cell'      : True,
                                   'cell_ref_form' : 'p',
                                   'cell_ref_kind' : 'ang',
                                   'cell_ref_tol'  : cell_ref_tol}
                nneg_ang_err_ind = amr.nneg_err_ang(mesh, uh_proj,
                                                    **kwargs_ang_nneg)
                        
                # If nneg error exceeds tolerance, we refine whichever
                # has larger negative values in columns to be refined.
                if ((nneg_spt_err_ind.col_max_err < col_ref_tol)
                    or (nneg_ang_err_ind.cell_max_err < col_ref_tol)):
                    # Use col_ref_tol here because otherwise this
                    # would always trigger.
                    err_ind_ang = nneg_ang_err_ind
                    err_ind_spt = nneg_spt_err_ind

                    avg_cell_ref_err = nneg_ang_err_ind.avg_cell_ref_err
                    avg_col_ref_err  = nneg_spt_err_ind.avg_col_ref_err
                    if np.abs(avg_cell_ref_err) > np.abs(avg_col_ref_err):
                        err_ind = nneg_ang_err_ind
                        ref_str = 'ang_nneg'
                    else:
                        err_ind = nneg_spt_err_ind
                        ref_str = 'spt_nneg'
                                
                else:
                    # If not using nneg err, then we use the jump error
                    kwargs_spt_jmp  = ref_strat['kwargs_spt_jmp']
                    jmp_spt_err_ind = amr.col_jump_err(mesh, uh_proj,
                                                       **kwargs_spt_jmp)
                    cell_ref_tol = -1. # Want uniform refinement
                    kwargs_ang_jmp = {'ref_col'      : False,
                                      'col_ref_form' : None,
                                      'col_ref_kind' : None,
                                      'col_ref_tol'  : None,
                                      'ref_cell'      : True,
                                      'cell_ref_form' : 'p',
                                      'cell_ref_kind' : 'ang',
                                      'cell_ref_tol'  : cell_ref_tol}
                    jmp_ang_err_ind = amr.cell_jump_err(mesh, uh_proj,
                                                       **kwargs_ang_jmp)
                    msg = (
                        'Comparison of average refinement error:\n' +
                        12 * ' ' + 'Angular {:.4E} '.format(jmp_ang_err_ind.avg_cell_ref_err) +
                        'vs. Spatial {:.4E}\n'.format(jmp_spt_err_ind.avg_col_ref_err)
                    )
                    utils.print_msg(msg, blocking = False)
                    err_ind_ang = jmp_ang_err_ind
                    err_ind_spt = jmp_spt_err_ind

                    avg_cell_ref_err = jmp_ang_err_ind.avg_cell_ref_err
                    avg_col_ref_err  = jmp_spt_err_ind.avg_col_ref_err
                            
                    # Randomly choose which, but probability depends on steering criterion
                    p_ang = avg_cell_ref_err / (avg_cell_ref_err + avg_col_ref_err)
                    ref_strs = ['ang_jmp', 'spt_jmp']
                    ref_str = rng.choice(ref_strs, size = 1, p = (p_ang, 1 - p_ang))[0]
                    #if p_ang >= 0.5:
                    #    ref_str = ref_strs[0]
                    #else:
                    #    ref_str = ref_strs[1]
                    if ref_str == 'ang_jmp':
                        err_ind = jmp_ang_err_ind
                    else: # ref_str == 'spt_jmp'
                        err_ind = jmp_spt_err_ind
                if do_plot_err_ind and (trial%10 == 0 or do_calc_err):
                    gen_err_ind_plot(mesh, err_ind_ang, trial, trial_dir, 'err_ind_ang.png')
                    gen_err_ind_plot(mesh, err_ind_spt, trial, trial_dir, 'err_ind_spt.png')
                avg_cell_ref_err_str = '{:.4E}'.format(jmp_ang_err_ind.avg_cell_ref_err)
                avg_col_ref_err_str = '{:.4E}'.format(jmp_spt_err_ind.avg_col_ref_err)
                refs += [[ndof, ref_str, avg_cell_ref_err_str, avg_col_ref_err_str, info]]
            else: # Just refining in space
                # Calculate nneg errors
                kwargs_spt_nneg  = ref_strat['kwargs_spt_nneg']
                col_ref_tol      = kwargs_spt_nneg['col_ref_tol']
                nneg_spt_err_ind = amr.nneg_err_spt(mesh, uh_proj,
                                                    **kwargs_spt_nneg)
                
                # If nneg error exceeds tolerance, we refine whichever
                # has larger negative values in columns to be refined.
                if (nneg_spt_err_ind.col_max_err < col_ref_tol):
                    err_ind = nneg_spt_err_ind
                    ref_str = 'spt_nneg'
                            
                else:
                    # If not using nneg err, then we use the jump error
                    kwargs_spt_jmp  = ref_strat['kwargs_spt_jmp']
                    jmp_spt_err_ind = amr.col_jump_err(mesh, uh_proj,
                                                       **kwargs_spt_jmp)
                    err_ind = jmp_spt_err_ind
                    ref_str = 'spt_jmp'
                if do_plot_err_ind and (trial%10 == 0 or do_calc_err):
                    gen_err_ind_plot(mesh, err_ind, trial, trial_dir, 'err_ind_spt.png')
                refs += [[ndof, ref_str, info]]
                    
            file_name = 'refs.txt'
            file_path = os.path.join(ref_strat_dir, file_name)
            json.dump(refs, open(file_path, 'w'))
                    
            msg = (
                '[Trial {}] Refining cause: {}\n'.format(trial, ref_str)
                )
            utils.print_msg(msg, blocking = False)
            mesh = amr.ref_by_ind(mesh, err_ind)
                    
    elif ((ref_strat['short_name'] == 'h-amr-all')
          or (ref_strat['short_name'] == 'p-amr-all')
          or (ref_strat['short_name'] == 'hp-amr-all')):
        if comm_rank == 0:
            # Have two choices: Refine based on nneg or jump, and if
            # also refining in space
                    
            # Calculate nneg errors
            kwargs_spt_nneg  = ref_strat['kwargs_spt_nneg']
            col_ref_tol      = kwargs_spt_nneg['col_ref_tol']
            nneg_spt_err_ind = amr.nneg_err_spt(mesh, uh_proj,
                                                **kwargs_spt_nneg)
                    
            kwargs_ang_nneg  = ref_strat['kwargs_ang_nneg']
            cell_ref_tol     = kwargs_ang_nneg['cell_ref_tol']
            nneg_ang_err_ind = amr.nneg_err_ang(mesh, uh_proj,
                                                **kwargs_ang_nneg)
                    
            # If nneg error exceeds tolerance, we refine whichever
            # has larger negative values in columns to be refined.
            if ((nneg_ang_err_ind.cell_max_err < cell_ref_tol)
                or (nneg_spt_err_ind.col_max_err < cell_ref_tol)):
                # Use cell_ref_tol here because otherwise this
                # would always trigger.
                err_ind_ang = nneg_ang_err_ind
                err_ind_spt = nneg_spt_err_ind

                avg_cell_ref_err = nneg_ang_err_ind.avg_cell_ref_err
                avg_col_ref_err  = nneg_spt_err_ind.avg_col_ref_err
                if np.abs(avg_cell_ref_err) > np.abs(avg_col_ref_err):
                    err_ind = nneg_ang_err_ind
                    ref_str = 'ang_nneg'
                else:
                    err_ind = nneg_spt_err_ind
                    ref_str = 'spt_nneg'
                            
            else:
                # If not using nneg err, then we use the jump error
                kwargs_ang_jmp  = ref_strat['kwargs_ang_jmp']
                jmp_ang_err_ind = amr.cell_jump_err(mesh, uh_proj,
                                                    **kwargs_ang_jmp)
                kwargs_spt_jmp  = ref_strat['kwargs_spt_jmp']
                jmp_spt_err_ind = amr.col_jump_err(mesh, uh_proj,
                                                   **kwargs_spt_jmp)
                msg = (
                    'Comparison of average refinement error:\n' +
                    12 * ' ' + 'Angular {:.4E} '.format(jmp_ang_err_ind.avg_cell_ref_err) +
                    'vs. Spatial {:.4E}\n'.format(jmp_spt_err_ind.avg_col_ref_err)
                )
                utils.print_msg(msg, blocking = False)
                err_ind_ang = jmp_ang_err_ind
                err_ind_spt = jmp_spt_err_ind

                avg_cell_ref_err = jmp_ang_err_ind.avg_cell_ref_err
                avg_col_ref_err  = jmp_spt_err_ind.avg_col_ref_err
                            
                # Randomly choose which, but probability depends on steering criterion
                p_ang = avg_cell_ref_err / (avg_cell_ref_err + avg_col_ref_err)
                ref_strs = ['ang_jmp', 'spt_jmp']
                ref_str = rng.choice(ref_strs, size = 1, p = (p_ang, 1 - p_ang))[0]
                #if p_ang >= 0.5:
                #    ref_str = ref_strs[0]
                #else:
                #    ref_str = ref_strs[1]
                if ref_str == 'ang_jmp':
                    err_ind = jmp_ang_err_ind
                else: # ref_str == 'spt_jmp'
                    err_ind = jmp_spt_err_ind
            if do_plot_err_ind and (trial%10 == 0 or do_calc_err):
                gen_err_ind_plot(mesh, err_ind_ang, trial, trial_dir, 'err_ind_ang.png')
                gen_err_ind_plot(mesh, err_ind_spt, trial, trial_dir, 'err_ind_spt.png')
            avg_cell_ref_err_str = '{:.4E}'.format(jmp_ang_err_ind.avg_cell_ref_err)
            avg_col_ref_err_str = '{:.4E}'.format(jmp_spt_err_ind.avg_col_ref_err)
            refs += [[ndof, ref_str, avg_cell_ref_err_str, avg_col_ref_err_str, info]]
                    
            file_name = 'refs.txt'
            file_path = os.path.join(ref_strat_dir, file_name)
            json.dump(refs, open(file_path, 'w'))
            
            msg = (
                '[Trial {}] Refining cause: {}\n'.format(trial, ref_str)
                )
            utils.print_msg(msg, blocking = False)
            mesh = amr.ref_by_ind(mesh, err_ind)

    try:
        err_ind
    except NameError:
        err_ind = None

    return [mesh, err_ind, refs]