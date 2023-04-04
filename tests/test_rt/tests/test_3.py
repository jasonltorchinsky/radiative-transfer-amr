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
from dg.mesh.utils import plot_mesh
from dg.matrix import get_intr_mask, split_matrix
from dg.projection import push_forward
import dg.quadrature as qd
from rt import calc_mass_matrix, calc_scat_matrix

from utils import print_msg

def test_3(dir_name = 'test_rt'):
    """
    Creates various visualizations of the scattering matrix and solves a 
    manufactured problem.
    """
    
    test_dir = os.path.join(dir_name, 'test_3')
    os.makedirs(test_dir, exist_ok = True)
    
    # Get the base mesh, manufactured solution
    [Lx, Ly]                   = [3., 2.]
    pbcs                       = [False, False]
    [ndof_x, ndof_y, ndof_th]  = [2, 2, 2]
    has_th                     = True
    mesh = gen_mesh(Ls     = [Lx, Ly],
                    pbcs   = pbcs,
                    ndofs  = [ndof_x, ndof_y, ndof_th],
                    has_th = has_th)
    
    [anl_sol, kappa, sigma, Phi, f] = get_cons_soln(prob_name = 'scat',
                                                    sol_num   = 2)
    
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
        plot_mesh(mesh,
                  file_name = os.path.join(trial_dir, 'mesh_3d.png'),
                  plot_dim  = 3)
        plot_mesh(mesh,
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

        ## Forcing vector, analytic solution, interior DOFs mask
        f_vec       = get_forcing_vec(mesh, f)
        anl_sol_vec = get_projection_vec(mesh, anl_sol)
        
        intr_mask        = get_intr_mask(mesh)
        f_vec_intr       = f_vec[intr_mask]
        anl_sol_vec_intr = anl_sol_vec[intr_mask]
        bcs_vec          = anl_sol_vec[np.invert(intr_mask)]
        
        ## Solve manufactured problem
        perf_soln_0 = perf_counter()
        print_msg('[Trial {}] Solving manufactured problem...'.format(trial))

        [M_intr, M_bdry] = split_matrix(mesh, M_mass - M_scat, intr_mask)

        apr_sol_vec_intr = spsolve(M_intr, f_vec_intr - M_bdry @ bcs_vec)
        
        perf_soln_f    = perf_counter()
        perf_soln_diff = perf_soln_f - perf_soln_0
        msg = (
            '[Trial {}] Manufactured problem solved! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]'.format(perf_cons_diff)
        )
        print_msg(msg)

        # Calculate eigenvalues of interior scattering matrix
        '''
        perf_evals_0 = perf_counter()
        print_msg('[Trial {}] Calculating eigenvalues...'.format(trial))
        
        M_intr_evals = np.linalg.eig(M_intr.toarray())[0]
        M_intr_evals = sorted(np.real(M_intr_evals), reverse = True)
        
        perf_evals_f    = perf_counter()
        perf_evals_diff = perf_evals_f - perf_evals_0
        msg = (
            '[Trial {}] Eigenvalues calculated! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]'.format(perf_evals_diff)
        )
        print_msg(msg)
        '''
        
        # Plot global scattering matrix
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
        ax.spy(M_scat,
               marker     = 's',
               markersize = 0.2,
               color      = 'k')
        ax.set_title('Global Scattering Matrix')
        
        file_name = 'scat_matrix.png'
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(os.path.join(trial_dir, file_name), dpi = 300)
        plt.close(fig)
        
        # Plot eigenvalues of interior scattering matrix
        '''
        fig, ax      = plt.subplots()
        ndof_intr    = np.shape(M_intr)[0]
        xx           = np.arange(1, ndof_intr + 1)
        
        ax.axhline(y         = 0.0,
                   color     = 'gray',
                   linestyle = '--',
                   linewidth = 0.1)
        ax.scatter(xx, M_intr_evals,
                   color = 'k',
                   s     = 0.15)

        ax.set_title('Interior Scattering Matrix - Eigenvalues (Real Part)')
        
        file_name = 'scat_matrix_evals.png'
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(os.path.join(trial_dir, file_name), dpi = 300)
        plt.close(fig)
        '''
        
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
        #mesh.ref_col(col_keys[-1], kind = 'ang')
        mesh.ref_mesh(kind = 'ang')

        perf_trial_f    = perf_counter()
        perf_trial_diff = perf_trial_f - perf_trial_0
        msg = (
            '[Trial {}] Completed! '.format(trial) +
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
    
    ax.set_xlabel('Total Degrees of Freedom')
    ax.set_ylabel('L$^{\infty}$ Error')
    
    ax.set_title('Uniform $h$-Refinement Convergence Rate - Scattering Problem')
    
    file_name = 'h-convergence-scatttering.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(test_dir, file_name), dpi = 300)
    plt.close(fig)
