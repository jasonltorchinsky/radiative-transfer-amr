import copy
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import os, sys

from .gen_mesh    import gen_mesh

sys.path.append('../../tests')
from test_cases import get_test_prob

sys.path.append('../../src')
from dg.mesh import get_hasnt_th
from dg.mesh.utils import plot_mesh
from dg.matrix import get_intr_mask, split_matrix, merge_vectors
from dg.projection import Projection, push_forward, to_projection, intg_th
from dg.projection.utils import plot_projection, plot_angular_dists
import dg.quadrature as qd
from rt import rtdg
from amr import anl_err, col_jump_err, cell_jump_err, ref_by_ind
from amr.utils import plot_error_indicator

from utils import print_msg


def test_3(dir_name = 'test_amr'):
    """
    Test inter-column jump error AMR
    """
    
    test_dir = os.path.join(dir_name, 'test_3')
    os.makedirs(test_dir, exist_ok = True)

    # Set the refinement type: 'sin' - single column
    #                        : 'uni' - uniform
    #                        : 'amr' - adaptive
    ref_type = 'amr'
    ntrial   = 9
    tol      = 0.8
    
    # Get the base mesh, test_problem
    [Lx, Ly]                   = [2., 3.]
    pbcs                       = [True, False]
    [ndof_x, ndof_y, ndof_th]  = [2, 2, 4]
    has_th                     = True
    mesh = gen_mesh(Ls     = [Lx, Ly],
                    pbcs   = pbcs,
                    ndofs  = [ndof_x, ndof_y, ndof_th],
                    has_th = has_th)
    
    [kappa, sigma, Phi, [bcs, dirac], f] = get_test_prob(prob_num = 1,
                                                         mesh = mesh)
    
    # Solve simplified problem over several trials
    ref_ndofs = np.zeros([ntrial])
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
        mesh_2d = get_hasnt_th(mesh)
        
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
        
        # Solve the test problem
        perf_cons_0 = perf_counter()
        print_msg('[Trial {}] Solving the test problem...'.format(trial))
        
        uh_proj = rtdg(mesh, kappa, sigma, Phi, [bcs, dirac], f)
        
        perf_cons_f    = perf_counter()
        perf_cons_diff = perf_cons_f - perf_cons_0
        msg = (
            '[Trial {}] Test problem solved! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]'.format(perf_cons_diff)
        )
        print_msg(msg)

        # Get number of DOFs
        uh_vec = uh_proj.to_vector()
        ref_ndofs[trial] = np.size(uh_vec)

        # Plot the solution
        file_name = os.path.join(trial_dir, 'uh.png')
        angles = np.linspace(0, 1.75, 8) * np.pi
        plot_projection(mesh, uh_proj, file_name = file_name, angles = angles)

        file_name = os.path.join(trial_dir, 'uh_slices.png')
        plot_angular_dists(mesh, uh_proj, file_name = file_name)

        mean_uh = intg_th(mesh, uh_proj)
        file_name = os.path.join(trial_dir, 'mean_uh.png')
        plot_projection(mesh_2d, mean_uh, file_name = file_name)

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
                   whis = [0, 100 * tol])

        ax.tick_params(
            axis      = 'y',    # changes apply to the y-axis
            which     = 'both', # both major and minor ticks are affected
            left      = False,  # ticks along the bottom edge are off
            right     = False,  # ticks along the top edge are off
            labelleft = False)  # labels along the bottom edge are off
        
        ax.set_xscale('log', base = 2)
        
        #xmin = 2**(np.floor(np.log2(min(col_jump_err_vals))))
        #xmax = 2**(np.ceil(np.log2(max(col_jump_err_vals))))
        #ax.set_xlim([xmin, xmax])
            
        ax.set_xlabel('Inter-Column Jump Error')

        yy = np.random.normal(1, 0.04, size = len(col_jump_err_vals))
        ax.plot(col_jump_err_vals, yy, 'k.', alpha = 0.8)
            
        file_name = 'col_jump_errs_dist.png'
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
            mesh.ref_mesh(kind = 'all')
        elif ref_type == 'amr':
            if (trial%3 == 0):
                mesh = ref_by_ind(mesh, col_jump_err_ind, tol)
            else:
                cell_jump_err_ind = cell_jump_err(mesh, uh_proj)
                mesh = ref_by_ind(mesh, cell_jump_err_ind, tol)
            
        perf_trial_f    = perf_counter()
        perf_trial_diff = perf_trial_f - perf_trial_0
        msg = (
            '[Trial {}] Trial completed! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]\n'.format(perf_trial_diff)
        )
        print_msg(msg)
        
    # Plot errors
    fig, ax = plt.subplots()

    ax.plot(ref_ndofs, jump_errs,
            label     = 'Inter-Column Jump Error',
            color     = 'r',
            linestyle = '-')

    ax.legend()

    ax.set_xscale('log', base = 2)
    ax.set_yscale('log', base = 2)

    max_err = max(jump_errs[1:])
    min_err = min(jump_errs[1:])
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
