import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve, eigs
from time import perf_counter
import os, sys

from .gen_mesh import gen_mesh

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
from amr import anl_err, ref_by_ind
from amr.utils import plot_error_indicator

from utils import print_msg


def test_5(dir_name = 'test_rt'):
    """
    Solves a manufactured problem of the full system.
    """
    
    test_dir = os.path.join(dir_name, 'test_5')
    os.makedirs(test_dir, exist_ok = True)

    # Set the refinement type: 'sin' - single column
    #                        : 'uni' - uniform
    #                        : 'amr' - adaptive
    ref_type = 'uni'
    ntrial   = 4
    
    # Get the base mesh, manufactured solution
    [Lx, Ly]                   = [2., 3.]
    pbcs                       = [False, False]
    [ndof_x, ndof_y, ndof_th]  = [2, 2, 2]
    has_th                     = True
    mesh = gen_mesh(Ls     = [Lx, Ly],
                    pbcs   = pbcs,
                    ndofs  = [ndof_x, ndof_y, ndof_th],
                    has_th = has_th)
    
    [u, kappa, sigma, Phi, f, _] = get_cons_prob(prob_name = 'comp',
                                                 prob_num  = 2,
                                                 mesh      = mesh)
    
    # Solve simplified problem over several trials
    ref_ndofs = np.zeros([ntrial])
    inf_errs  = np.zeros([ntrial])
    for trial in range(0, ntrial):
        perf_trial_0 = perf_counter()
        print_msg('[Trial {}] Starting...\n'.format(trial))
            
        # Set up output directories
        trial_dir = os.path.join(test_dir, 'trial_{}'.format(trial))
        os.makedirs(trial_dir, exist_ok = True)

        # Plot the mesh
        plot_mesh(mesh,
                  file_name = os.path.join(trial_dir, 'mesh_3d.png'),
                  plot_dim  = 3)
        plot_mesh(mesh,
                  file_name   = os.path.join(trial_dir, 'mesh_2d.png'),
                  plot_dim    = 2,
                  label_cells = (trial <= 3))

        # Get the ending indices for the column matrices, number of DOFs in mesh
        col_items = sorted(mesh.cols.items())
        ncol      = 0
        for col_key, col in col_items:
            if col.is_lf:
                ncol += 1
        
        col_end_idxs = [0] * ncol
        mesh_ndof    = 0
        idx          = 0
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
                col_end_idxs[idx] = mesh_ndof
                
                idx += 1
                
        ref_ndofs[trial] = mesh_ndof
        
        # Construct matrices to solve manufactured problem
        ## Mass matrix
        perf_cons_0 = perf_counter()
        print_msg('[Trial {}] Constructing mass matrix...'.format(trial))
        
        M_mass = calc_mass_matrix(mesh, kappa)
        
        perf_cons_f    = perf_counter()
        perf_cons_diff = perf_cons_f - perf_cons_0
        msg = (
            '[Trial {}] Mass matrix constructed! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]'.format(perf_cons_diff)
        )
        print_msg(msg)
        
        ## Scattering matrix
        perf_cons_0 = perf_counter()
        print_msg('[Trial {}] Constructing scattering matrix...'.format(trial))
        
        M_scat = calc_scat_matrix(mesh, sigma, Phi)
        
        perf_cons_f    = perf_counter()
        perf_cons_diff = perf_cons_f - perf_cons_0
        msg = (
            '[Trial {}] Scattering matrix constructed! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]'.format(perf_cons_diff)
        )
        print_msg(msg)
        
        ## Interior convection matrix
        perf_cons_0 = perf_counter()
        print_msg('[Trial {}] Constructing interior convection matrix...'.format(trial))
        
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
        print_msg('[Trial {}] Constructing boundary convection matrix...'.format(trial))
        
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
        print_msg('[Trial {}] Solving manufactured problem...'.format(trial))
        
        M = (M_bdry_conv - M_intr_conv) + M_mass - M_scat
        [M_intr, M_bdry] = split_matrix(mesh, M, intr_mask)
        
        uh_vec_intr = spsolve(M_intr, f_vec_intr - M_bdry @ bcs_vec)
        
        perf_soln_f    = perf_counter()
        perf_soln_diff = perf_soln_f - perf_soln_0
        msg = (
            '[Trial {}] Manufactured problem solved! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]'.format(perf_soln_diff)
        )
        print_msg(msg)

        # Plot the analytic, approximate solutions
        uh_vec = merge_vectors(uh_vec_intr, bcs_vec, intr_mask)
        uh_proj = to_projection(mesh, uh_vec)
        file_name = os.path.join(trial_dir, 'uh_proj.png')
        angles = [0, np.pi/3, 2 * np.pi / 3, np.pi,
                  4 * np.pi / 3, 5 * np.pi / 3]
        plot_projection(mesh, uh_proj, file_name = file_name, angles = angles)
        
        file_name = os.path.join(trial_dir, 'uh_ang_dist.png')
        plot_angular_dists(mesh, uh_proj, file_name = file_name)
        
        mesh_2d = get_hasnt_th(mesh)
        mean_uh = intg_th(mesh, uh_proj)
        file_name = os.path.join(trial_dir, 'uh_mean.png')
        plot_projection(mesh_2d, mean_uh, file_name = file_name)

        file_name = os.path.join(trial_dir, 'u_proj.png')
        angles = [0, np.pi/3, 2 * np.pi / 3, np.pi,
                  4 * np.pi / 3, 5 * np.pi / 3]
        plot_projection(mesh, u_proj, file_name = file_name, angles = angles)
        
        file_name = os.path.join(trial_dir, 'u_ang_dist.png')
        plot_angular_dists(mesh, u_proj, file_name = file_name)
        
        mean_u = intg_th(mesh, u_proj)
        file_name = os.path.join(trial_dir, 'u_mean.png')
        plot_projection(mesh_2d, mean_u, file_name = file_name)
        
        # Plot the difference in solutions
        diff_vec_intr = uh_vec_intr - u_vec_intr
        zero_bcs_vec  = 0. * bcs_vec
        diff_vec      = merge_vectors(diff_vec_intr, zero_bcs_vec, intr_mask)
        diff_proj     = to_projection(mesh, diff_vec)
        
        file_name = os.path.join(trial_dir, 'diff.png')
        angles = [0, np.pi/3, 2 * np.pi / 3, np.pi,
                  4 * np.pi / 3, 5 * np.pi / 3]
        plot_projection(mesh, diff_proj, file_name = file_name, angles = angles)

        # Plot the analytic error indicator
        anl_err_ind = anl_err(mesh, uh_proj, u)
        file_name = os.path.join(trial_dir, 'anl_errs.png')
        plot_error_indicator(mesh, anl_err_ind, file_name = file_name,
                             name = 'Analytic Max-Norm Column')

        
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
        ax.set_title('Global Complete Matrix')
        
        file_name = 'comp_matrix.png'
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(os.path.join(trial_dir, file_name), dpi = 300)
        plt.close(fig)
        
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
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(os.path.join(trial_dir, file_name), dpi = 300)
        plt.close(fig)
        
        # Caluclate error
        inf_errs[trial] = np.amax(np.abs(u_vec_intr - uh_vec_intr)) \
            / np.amax(np.abs(u_vec_intr))

        # Refine the mesh for the next trial
        if ref_type == 'sin':
            ## Refine a given column
            col_keys = sorted(mesh.cols.keys())
            mesh.ref_col(col_keys[-4], kind = 'all')
        elif ref_type == 'uni':
            ## Refine the mesh uniformly
            mesh.ref_mesh(kind = 'ang')
        elif ref_type == 'amr':
            ## Refine the column spatially with the biggest "error"
            mesh = ref_by_ind(mesh, anl_err_ind, 0.9)

        perf_trial_f    = perf_counter()
        perf_trial_diff = perf_trial_f - perf_trial_0
        msg = (
            '[Trial {}] Trial completed! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]\n'.format(perf_trial_diff)
        )
        print_msg(msg)
        
    # Plot errors
    fig, ax = plt.subplots()
    
    ax.plot(ref_ndofs, inf_errs,
            label     = 'L$^{\infty}$ Error',
            color     = 'k',
            linestyle = '-')

    ax.set_xscale('log', base = 2)
    ax.set_yscale('log', base = 2)

    if np.log2(max(inf_errs)) - np.log2(min(inf_errs)) < 1:
        ymin = 2**(np.floor(np.log2(min(inf_errs))))
        ymax = 2**(np.ceil(np.log2(max(inf_errs))))
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
    title_str = '{} $h$-Refinement Convergence Rate - Complete Problem'.format(ref_str)
    ax.set_title(title_str)
    
    file_name = 'h-convergence-complete.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(test_dir, file_name), dpi = 300)
    plt.close(fig)
