import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve, eigs
from time import perf_counter
import os, sys

from .gen_mesh           import gen_mesh
from .get_forcing_vec    import get_forcing_vec
from .get_projection_vec import get_projection_vec
from .get_cons_soln      import get_cons_soln

sys.path.append('../../src')
from dg.mesh import ji_mesh
from dg.mesh import tools as mesh_tools
import dg.quadrature as qd
from rad_amr import calc_mass_matrix, calc_scat_matrix, \
    calc_intr_conv_matrix, calc_bdry_conv_matrix, \
    get_intr_mask, split_matrix

from utils import print_msg


def test_5(dir_name = 'test_rtdg'):
    """
    Solves a manufactured problem of the full system.
    """
    
    test_dir = os.path.join(dir_name, 'test_5')
    os.makedirs(test_dir, exist_ok = True)
    
    # Get the base mesh, manufactured solution
    [Lx, Ly]                   = [3., 2.]
    pbcs                       = [False, False]
    [ndof_x, ndof_y, ndof_th]  = [4, 4, 4]
    has_th                     = True
    mesh = gen_mesh(Ls     = [Lx, Ly],
                    pbcs   = pbcs,
                    ndofs  = [ndof_x, ndof_y, ndof_th],
                    has_th = has_th)
    
    [anl_sol, kappa, sigma, Phi, f] = get_cons_soln(prob_name = 'comp',
                                                    sol_num   = 0)
    
    # Solve simplified problem over several trials
    ntrial    = 3
    ref_ndofs = np.zeros([ntrial])
    inf_errs  = np.zeros([ntrial])
    for trial in range(0, ntrial):
        perf_trial_0 = perf_counter()
        print_msg('[Trial {}] Starting...'.format(trial))
            
        # Set up output directories
        trial_dir = os.path.join(test_dir, 'trial_{}'.format(trial))
        os.makedirs(trial_dir, exist_ok = True)

        # Plot the mesh
        mesh_tools.plot_mesh(mesh,
                             file_name = os.path.join(trial_dir, 'mesh_3d.png'),
                             plot_dim  = 3)
        mesh_tools.plot_mesh(mesh,
                             file_name   = os.path.join(trial_dir, 'mesh_2d.png'),
                             plot_dim    = 2,
                             label_cells = True)

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
        f_vec       = get_forcing_vec(mesh, f)
        anl_sol_vec = get_projection_vec(mesh, anl_sol)
        
        intr_mask        = get_intr_mask(mesh)
        bdry_mask        = np.invert(intr_mask)
        f_vec_intr       = f_vec[intr_mask]
        anl_sol_vec_intr = anl_sol_vec[intr_mask]
        bcs_vec          = anl_sol_vec[bdry_mask]
        
        ## Solve manufactured problem
        perf_soln_0 = perf_counter()
        print_msg('[Trial {}] Solving manufactured problem...'.format(trial))
        
        M = (M_bdry_conv - M_intr_conv) + M_mass - M_scat
        [M_intr, M_bdry] = split_matrix(mesh, M)
        
        apr_sol_vec_intr = spsolve(M_intr, f_vec_intr - M_bdry @ bcs_vec)
        
        perf_soln_f    = perf_counter()
        perf_soln_diff = perf_soln_f - perf_soln_0
        msg = (
            '[Trial {}] Manufactured problem solved! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]'.format(perf_cons_diff)
        )
        print_msg(msg)
        
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
        
        file_name = 'conv_matrix.png'
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(os.path.join(trial_dir, file_name), dpi = 300)
        plt.close(fig)

        
        # Plot solutions
        fig, ax = plt.subplots()

        ax.plot(anl_sol_vec_intr,
                label = 'Analytic Solution',
                color = 'r',
                drawstyle = 'steps-post')
        ax.plot(apr_sol_vec_intr,
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
        inf_errs[trial] = np.amax(np.abs(anl_sol_vec_intr - apr_sol_vec_intr))

        # Refine the mesh for the next trial
        #col_keys = sorted(mesh.cols.keys())
        #mesh.ref_col(col_keys[-1], kind = 'all')
        mesh.ref_mesh(kind = 'all')

        perf_trial_f    = perf_counter()
        perf_trial_diff = perf_trial_f - perf_trial_0
        msg = (
            '[Trial {}] Completed! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]'.format(perf_trial_diff)
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
    
    ax.set_title('Uniform $h$-Refinement Convergence Rate - Complete Problem')
    
    file_name = 'h-convergence-complete.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(test_dir, file_name), dpi = 300)
    plt.close(fig)
