import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from time import perf_counter
import os, sys

from .gen_mesh  import gen_mesh

sys.path.append('../../tests')
from test_cases import get_cons_prob

sys.path.append('../../src')
from dg.mesh.utils import plot_mesh
from dg.matrix import get_intr_mask, split_matrix, merge_vectors
from dg.projection import Projection, push_forward, to_projection
from dg.projection.utils import plot_projection, plot_angular_dists
import dg.quadrature as qd
from rt import calc_mass_matrix, calc_scat_matrix, \
    calc_intr_conv_matrix, calc_bdry_conv_matrix, \
    calc_forcing_vec
from amr import anl_err, col_jump_err, cell_jump_err, ref_by_ind
from amr.utils import plot_error_indicator

from utils import print_msg


def test_2(dir_name = 'test_amr'):
    """
    Tests the jump error calculation.
    """
    
    test_dir = os.path.join(dir_name, 'test_2')
    os.makedirs(test_dir, exist_ok = True)

    # Set the refinement type: 'sin' - single column
    #                        : 'uni' - uniform
    #                        : 'amr' - adaptive
    ref_type = 'amr'
    ntrial   = 5
    tol      = 0.8
    
    # Get the base mesh, test_problem
    [Lx, Ly]                   = [3., 2.]
    pbcs                       = [False, False]
    [ndof_x, ndof_y, ndof_th]  = [2, 2, 2]
    has_th                     = True
    mesh = gen_mesh(Ls     = [Lx, Ly],
                    pbcs   = pbcs,
                    ndofs  = [ndof_x, ndof_y, ndof_th],
                    has_th = has_th)
    
    [anl_sol, kappa, sigma, Phi, f, _] = get_cons_prob(prob_name = 'comp',
                                                       prob_num  = 2)
    
    # Solve simplified problem over several trials
    ref_ndofs = np.zeros([ntrial])
    sol_errs  = np.zeros([ntrial])
    jump_errs = np.zeros([ntrial])
    for trial in range(0, ntrial):
        perf_trial_0 = perf_counter()
        print_msg('[Trial {}] Starting...'.format(trial))
            
        # Set up output directories
        trial_dir = os.path.join(test_dir, 'trial_{}'.format(trial))
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

        # Plot the coefficient functions
        
        # Construct solve the test problem
        perf_cons_0 = perf_counter()
        print_msg('[Trial {}] Solving the test problem...'.format(trial))

        # Use the analytic solution for boundary conditions
        M_mass = calc_mass_matrix(mesh, kappa)
        M_scat = calc_scat_matrix(mesh, sigma, Phi)
        M_intr_conv = calc_intr_conv_matrix(mesh)
        M_bdry_conv = calc_bdry_conv_matrix(mesh)

        f_vec       = calc_forcing_vec(mesh, f)

        u_proj = Projection(mesh, anl_sol)
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
        
        perf_cons_f    = perf_counter()
        perf_cons_diff = perf_cons_f - perf_cons_0
        msg = (
            '[Trial {}] Test problem solved! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]'.format(perf_cons_diff)
        )
        print_msg(msg)

        # Get number of DOFs
        ref_ndofs[trial] = np.size(f_vec)
        
        # Calculate the maximum error of the solution
        diff_vec = np.abs(uh_vec - u_vec)
        max_err  = np.amax(diff_vec)

        sol_errs[trial] = max_err

        # Plot the numerial solution
        file_name = os.path.join(trial_dir, 'uh.png')
        angles = np.linspace(0, 1.75, 8) * np.pi
        plot_projection(mesh, uh_proj, file_name = file_name, angles = angles)

        file_name = os.path.join(trial_dir, 'uh_slices.png')
        plot_angular_dists(mesh, uh_proj, file_name = file_name)

        # Plot the analytic solution
        file_name = os.path.join(trial_dir, 'u.png')
        angles = np.linspace(0, 1.75, 8) * np.pi
        plot_projection(mesh, u_proj, file_name = file_name, angles = angles)

        file_name = os.path.join(trial_dir, 'u_slices.png')
        plot_angular_dists(mesh, u_proj, file_name = file_name)

        # Plot the jump error indicator
        col_jump_err_ind = col_jump_err(mesh, uh_proj)
        file_name = os.path.join(trial_dir, 'col_jump_errs.png')
        plot_error_indicator(mesh, col_jump_err_ind, file_name = file_name,
                             name = 'Inter-Column Jump')

        col_jump_err_ind_dict = {}
        col_items = sorted(mesh.cols.items())
        for col_key, col in col_items:
            if col.is_lf:
                col_jump_err_ind_dict[col_key] = col_jump_err_ind.cols[col_key].err_ind
        col_jump_err_vals = list(col_jump_err_ind_dict.values())
        max_err = max(col_jump_err_vals)
        jump_errs[trial] = max_err
        
        fig, ax = plt.subplots()
    
        ax.boxplot(col_jump_err_vals,
                   vert = False,
                   whis = [0, 90])

        ax.tick_params(
            axis      = 'y',    # changes apply to the y-axis
            which     = 'both', # both major and minor ticks are affected
            left      = False,  # ticks along the bottom edge are off
            right     = False,  # ticks along the top edge are off
            labelleft = False)  # labels along the bottom edge are off
        
        ax.set_xscale('log', base = 2)
        
        xmin = 2**(np.floor(np.log2(min(col_jump_err_vals))))
        xmax = 2**(np.ceil(np.log2(max(col_jump_err_vals))))
        ax.set_xlim([xmin, xmax])
            
        ax.set_xlabel('Inter-Column Jump Error')

        yy = np.random.normal(1, 0.04, size = len(col_jump_err_vals))
        ax.plot(col_jump_err_vals, yy, 'k.', alpha = 0.8)
            
        file_name = 'col_jump_errs_dist.png'
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(os.path.join(trial_dir, file_name), dpi = 300)
        plt.close(fig)

        # Plot the analytic error indicator
        anl_err_ind = anl_err(mesh, uh_proj, anl_sol)
        file_name = os.path.join(trial_dir, 'anl_errs.png')
        plot_error_indicator(mesh, anl_err_ind, file_name = file_name,
                             name = 'Analytic Max-Norm Column')

        anl_err_ind_dict = {}
        col_items = sorted(mesh.cols.items())
        for col_key, col in col_items:
            if col.is_lf:
                anl_err_ind_dict[col_key] = anl_err_ind.cols[col_key].err_ind
        anl_err_vals = list(anl_err_ind_dict.values())
        
        fig, ax = plt.subplots()
    
        ax.boxplot(anl_err_vals,
                   vert = False,
                   whis = [0, 100 * tol])

        ax.tick_params(
            axis      = 'y',    # changes apply to the y-axis
            which     = 'both', # both major and minor ticks are affected
            left      = False,  # ticks along the bottom edge are off
            right     = False,  # ticks along the top edge are off
            labelleft = False)  # labels along the bottom edge are off
        
        ax.set_xscale('log', base = 2)
        
        xmin = 2**(np.floor(np.log2(min(anl_err_vals))))
        xmax = 2**(np.ceil(np.log2(max(anl_err_vals))))
        ax.set_xlim([xmin, xmax])
            
        ax.set_xlabel('Analytic Column Error')

        yy = np.random.normal(1, 0.04, size = len(anl_err_vals))
        ax.plot(anl_err_vals, yy, 'k.', alpha = 0.8)
            
        file_name = 'anl_errs_dist.png'
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(os.path.join(trial_dir, file_name), dpi = 300)
        plt.close(fig)
        
        # Refine the mesh for the next trial
        if ref_type == 'sin':
            ## Refine a given column
            col_keys = sorted(mesh.cols.keys())
            mesh.ref_col(col_keys[-4], kind = 'all')
        elif ref_type == 'uni':
            ## Refine the mesh uniformly
            mesh.ref_mesh(kind = 'spt')
        elif ref_type == 'amr':
            cell_jump_err_ind = cell_jump_err(mesh, uh_proj)
            mesh = ref_by_ind(mesh, cell_jump_err_ind, tol)
            mesh = ref_by_ind(mesh, col_jump_err_ind, tol)
            
        perf_trial_f    = perf_counter()
        perf_trial_diff = perf_trial_f - perf_trial_0
        msg = (
            '[Trial {}] Trial completed! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]\n'.format(perf_trial_diff)
        )
        print_msg(msg)
        
    # Plot errors
    fig, ax = plt.subplots()
    
    ax.plot(ref_ndofs, sol_errs,
            label     = 'L$^{\infty}$ Error',
            color     = 'k',
            linestyle = '-')

    ax.plot(ref_ndofs, jump_errs,
            label     = 'Inter-Column Jump Error',
            color     = 'r',
            linestyle = '-')

    ax.legend()

    ax.set_xscale('log', base = 2)
    ax.set_yscale('log', base = 2)

    max_err = max([max(jump_errs[1:]), max(sol_errs)])
    min_err = min([min(jump_errs[1:]), min(sol_errs)])
    ymin = 2**(np.floor(np.log2(min_err)))
    ymax = 2**(np.ceil(np.log2(max_err)))
    ax.set_ylim([ymin, ymax])
    
    ax.set_xlabel('Total Degrees of Freedom')
    ax.set_ylabel('L$^{\infty}$ Error')


    ref_str = ''
    if ref_type == 'sin':
        ref_str = 'Single Column'
    elif ref_type == 'uni':
        ref_str = 'Uniform'
    elif ref_type == 'amr':
        ref_str = 'Adaptive'
    title_str = '{} $h$-Refinement Solution Convergence Rate'.format(ref_str)
    ax.set_title(title_str)
    
    file_name = 'h-sol-convergence.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(test_dir, file_name), dpi = 300)
    plt.close(fig)
