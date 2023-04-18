import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from time import perf_counter
import os, sys

from .gen_mesh  import gen_mesh

sys.path.append('../../tests')
from test_cases import get_cons_prob

sys.path.append('../../src')
from dg.mesh import get_hasnt_th
from dg.mesh.utils import plot_mesh
from dg.matrix import get_intr_mask, split_matrix, merge_vectors
from dg.projection import Projection, push_forward, to_projection, intg_th
from dg.projection.utils import plot_projection, plot_angular_dists
import dg.quadrature as qd
from rt import calc_mass_matrix, calc_scat_matrix, \
    calc_intr_conv_matrix, calc_bdry_conv_matrix, \
    calc_forcing_vec
from amr import anl_err_spt, col_jump_err, ref_by_ind
from amr.utils import plot_error_indicator, plot_error_indicator_dist

from utils import print_msg


def test_2(dir_name = 'test_amr'):
    """
    Compare the inter-column jump error estimator against the analytic error
    estimator.
    """
    
    test_dir = os.path.join(dir_name, 'test_2')
    os.makedirs(test_dir, exist_ok = True)

    colors = ['#E69F00', '#56B4E9', '#009E73',
              '#F0E442', '#0072B2', '#D55E00',
              '#CC79A7']
    ncolor = len(colors)

    ref_types = ['uni-spt', 'amr-anl-spt', 'amr-jmp']
    nref_type = len(ref_types)
    
    max_ndof   = 2**13
    max_ntrial = 9
    tol        = 0.9
    
    # Set up arrays to store error
    ref_ndofs = {}
    errs  = {}

    for ref_type in ref_types:
        perf_ref_type_0 = perf_counter()
        print_msg('[Refinement Type - {}] Starting...'.format(ref_type))

        ref_dir = os.path.join(test_dir, ref_type)
        os.makedirs(test_dir, exist_ok = True)
    
        # Get the base mesh, test_problem
        [Lx, Ly]                   = [2., 3.]
        pbcs                       = [False, False]
        [ndof_x, ndof_y, ndof_th]  = [2, 2, 4]
        has_th                     = True
        mesh = gen_mesh(Ls     = [Lx, Ly],
                        pbcs   = pbcs,
                        ndofs  = [ndof_x, ndof_y, ndof_th],
                        has_th = has_th)
        
        [u, kappa, sigma, Phi, f, u_intg_th, _] = get_cons_prob(prob_name = 'comp',
                                                                prob_num  = 1,
                                                                mesh      = mesh)
        
        ref_ndofs[ref_type] = []
        errs[ref_type] = []
        ndof = 0
        trial = 0
        
        while (ndof < max_ndof) and (trial <= max_ntrial):
            perf_trial_0 = perf_counter()
            print_msg('[Trial {}] Starting...'.format(trial))
            
            # Set up output directories
            trial_dir = os.path.join(ref_dir, 'trial_{}'.format(trial))
            os.makedirs(trial_dir, exist_ok = True)
            
            # Plot the mesh
            file_name = os.path.join(trial_dir, 'mesh_3d.png')
            plot_mesh(mesh,
                      file_name = file_name,
                      plot_dim  = 3)
            
            file_name = os.path.join(trial_dir, 'mesh_2d.png')
            plot_mesh(mesh,
                      file_name   = file_name,
                      plot_dim    = 2,
                      label_cells = (trial <= 3))
            
            # Construct solve the test problem
            perf_sol_0 = perf_counter()
            print_msg('[Trial {}] Solving the test problem...'.format(trial))
            
            # Use the analytic solution for boundary conditions
            M_mass = calc_mass_matrix(mesh, kappa)
            M_scat = calc_scat_matrix(mesh, sigma, Phi)
            M_intr_conv = calc_intr_conv_matrix(mesh)
            M_bdry_conv = calc_bdry_conv_matrix(mesh)
            
            f_vec  = calc_forcing_vec(mesh, f)
            
            u_proj = Projection(mesh, u)
            u_vec  = u_proj.to_vector()
            
            intr_mask  = get_intr_mask(mesh)
            bdry_mask  = np.invert(intr_mask)
            f_vec_intr = f_vec[intr_mask]
            u_vec_intr = u_vec[intr_mask]
            bcs_vec    = u_vec[bdry_mask]
            
            M = (M_bdry_conv - M_intr_conv) + M_mass - M_scat
            [M_intr, M_bdry] = split_matrix(mesh, M, intr_mask)
            
            uh_vec_intr = spsolve(M_intr, f_vec_intr - M_bdry @ bcs_vec)
            uh_vec      = merge_vectors(uh_vec_intr, bcs_vec, intr_mask)
            uh_proj     = to_projection(mesh, uh_vec)
            
            perf_sol_f    = perf_counter()
            perf_sol_diff = perf_sol_f - perf_sol_0
            msg = (
                '[Trial {}] Test problem solved! '.format(trial) +
                'Time Elapsed: {:08.3f} [s]'.format(perf_sol_diff)
            )
            print_msg(msg)

            # Get number of DOFs
            ndof = np.size(f_vec)
            ref_ndofs[ref_type].append(ndof)

            msg = (
                '[Trial {}] Number of DOFs:'.format(trial) +
                ' {} of {}'.format(ndof, max_ndof)
            )
            print_msg(msg)
            
            # Calculate the maximum error of the solution
            max_err  = np.amax(np.abs(u_vec - uh_vec)) / np.amax(np.abs(u_vec))
            errs[ref_type].append(max_err)
            
            # Plot the numerial solution
            file_name = os.path.join(trial_dir, 'uh.png')
            angles = np.linspace(0, 1.75, 8) * np.pi
            plot_projection(mesh, uh_proj, file_name = file_name, angles = angles)
            
            mesh_2d = get_hasnt_th(mesh)
            mean_uh = intg_th(mesh, uh_proj)
            file_name = os.path.join(trial_dir, 'uh_mean.png')
            plot_projection(mesh_2d, mean_uh, file_name = file_name)
            
            file_name = os.path.join(trial_dir, 'uh_slices.png')
            plot_angular_dists(mesh, uh_proj, file_name = file_name)
            
            # Plot the analytic solution
            file_name = os.path.join(trial_dir, 'u.png')
            angles = np.linspace(0, 1.75, 8) * np.pi
            plot_projection(mesh, u_proj, file_name = file_name, angles = angles)
            
            file_name = os.path.join(trial_dir, 'u_slices.png')
            plot_angular_dists(mesh, u_proj, file_name = file_name)
            
            mean_u = intg_th(mesh, u_proj)
            file_name = os.path.join(trial_dir, 'u_mean.png')
            plot_projection(mesh_2d, mean_u, file_name = file_name)
            
            # Plot the jump error indicator
            col_jump_err_ind = col_jump_err(mesh, uh_proj)
            file_name = os.path.join(trial_dir, 'col_jump_err.png')
            plot_error_indicator(mesh, col_jump_err_ind, file_name = file_name,
                                 name = 'Inter-Column Jump')
            
            file_name = os.path.join(trial_dir, 'col_jump_err_dist.png')
            plot_error_indicator_dist(mesh, col_jump_err_ind, file_name = file_name,
                                      name = 'Inter-Column Jump')
            
            # Plot the analytic error (spatial) indicator
            anl_err_spt_ind = anl_err_spt(mesh, uh_proj, u_intg_th)
            file_name = os.path.join(trial_dir, 'anl_err_spt.png')
            plot_error_indicator(mesh, anl_err_spt_ind, file_name = file_name,
                                 name = 'Analytic (Spatial) Max-Norm Column')
            
            file_name = os.path.join(trial_dir, 'anl_err_spt_dist.png')
            plot_error_indicator_dist(mesh, anl_err_spt_ind, file_name = file_name,
                                      name = 'Analytic (Spatial) Max-Norm Column')
            
            # Refine the mesh for the next trial
            if ref_type == 'uni-spt':
                mesh.ref_mesh(kind = 'spt')
            elif ref_type == 'amr-anl-spt':
                anl_err_spt_ind = anl_err_spt(mesh, uh_proj, u_intg_th)
                mesh = ref_by_ind(mesh, anl_err_spt_ind, tol)
            elif ref_type == 'amr-jmp':
                col_jump_err_ind = col_jump_err(mesh, uh_proj)
                mesh = ref_by_ind(mesh, col_jump_err_ind, tol)
                
            perf_trial_f    = perf_counter()
            perf_trial_diff = perf_trial_f - perf_trial_0
            msg = (
                '[Trial {}] Trial completed! '.format(trial) +
                'Time Elapsed: {:08.3f} [s]\n'.format(perf_trial_diff)
            )
            print_msg(msg)
            
            trial += 1
            
        perf_ref_type_f = perf_counter()
        perf_ref_type_diff = perf_ref_type_f - perf_ref_type_0
        msg = (
            '[Refinement Type - {}] Refinement type trials completed! '.format(ref_type) +
            'Time Elapsed: {:08.3f} [s]\n'.format(perf_ref_type_diff)
        )
        print_msg(msg)
        
    # Plot errors
    fig, ax = plt.subplots()
    [min_err, max_err] = [10**10, -10**10]
    
    for rr in range(0, nref_type):
        ref_type = ref_types[rr]
        if ref_type == 'uni-spt':
            label = 'Uniform - Spatial'
        elif ref_type == 'amr-anl':
            label = 'Adaptive (Analytic) - Spatial'
        elif ref_type == 'amr-anl-spt':
            label = 'Adaptive (Analytic - Spatial) - Spatial'
        elif ref_type == 'amr-jmp':
            label = 'Adaptive (Inter-Column Jump) - Spatial'
            
        min_err = min([min_err, min(errs[ref_type])])
        max_err = max([max_err, max(errs[ref_type])])
        
        ax.plot(ref_ndofs[ref_type], errs[ref_type],
                label     = label,
                color     = colors[rr],
                linestyle = '-')
        
    ax.legend()
    
    ax.set_xscale('log', base = 2)
    ax.set_yscale('log', base = 2)
    
    ymin = 2**(np.floor(np.log2(min_err)))
    ymax = 2**(np.ceil(np.log2(max_err)))
    ax.set_ylim([ymin, ymax])
    
    ax.set_xlabel('Total Degrees of Freedom')
    ax.set_ylabel('L$^{\infty}$ Error')
    
    title_str = 'Spatial $h$-Refinement Solution Convergence Rate'
    ax.set_title(title_str)
    
    file_name = 'h-ref-conv.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(test_dir, file_name), dpi = 300)
    plt.close(fig)
