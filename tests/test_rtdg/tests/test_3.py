import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve, eigs
from time import perf_counter
import os, sys

from .get_forcing_vec    import get_forcing_vec
from .get_projection_vec import get_projection_vec

sys.path.append('../../src')
from dg.mesh import ji_mesh
from dg.mesh import tools as mesh_tools
import dg.quadrature as qd
from rad_amr import calc_intr_conv_matrix, calc_bdry_conv_matrix, \
    get_intr_mask, split_matrix

from utils import print_msg

# Utilize a manufactured solution
def anl_sol(x, y, th):
    # Also used to calculate BCs!
    return np.sin(th)**2 * np.exp(-(x**2 + y**2))

def f(x, y, th):
    return -2. * np.sin(th)**2 * np.exp(-(x**2 + y**2)) \
    * (x * np.cos(th) + y * np.sin(th))

def test_3(dir_name = 'test_rtdg'):
    """
    Creates various visualizations of the convection matrix and solves a 
    manufactured problem.
    """
    
    test_dir = os.path.join(dir_name, 'test_3')
    os.makedirs(test_dir, exist_ok = True)
    
    # Create the base mesh which will be refined in each trial.
    [Lx, Ly]                   = [3., 2.]
    [ndof_x, ndof_y, ndof_th]  = [8, 8, 8]
    mesh = ji_mesh.Mesh(Ls     = [Lx, Ly],
                        pbcs   = [False, False],
                        ndofs  = [ndof_x, ndof_y, ndof_th],
                        has_th = True)
    
    # Refine the mesh for initial trial
    for _ in range(0, 2):
        mesh.cols[0].ref_col()
    for _ in range(0, 1):
        mesh.ref_mesh()
    
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
        
        M_conv = M_bdry_conv - M_intr_conv
        [M_intr, M_bdry] = split_matrix(mesh, M_conv)
        
        apr_sol_vec_intr = spsolve(M_intr, f_vec_intr - M_bdry @ bcs_vec)
        
        perf_soln_f    = perf_counter()
        perf_soln_diff = perf_soln_f - perf_soln_0
        msg = (
            '[Trial {}] Manufactured problem solved! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]'.format(perf_cons_diff)
        )
        print_msg(msg)

        # Calculate eigenvalues of interior convection matrix
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
        
        # Plot global convection matrix
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
        ax.spy(M_conv,
               marker     = 's',
               markersize = 0.2,
               color      = 'k')
        ax.set_title('Global Convection Matrix')
        
        file_name = 'conv_matrix.png'
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(os.path.join(trial_dir, file_name), dpi = 300)
        plt.close(fig)
        
        # Plot eigenvalues of interior convection matrix
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

        ax.set_title('Interior Convection Matrix - Eigenvalues (Real Part)')
        
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
        #for col_key, col in col_items:
        #    if col.is_lf:
        #        col.ref_col()
        mesh.ref_mesh()

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
    
    ax.set_xlabel('Total Degrees of Freedom')
    ax.set_ylabel('L$^{\infty}$ Error')
    
    ax.set_title('Uniform $h$-Refinement Convergence Rate - Convection Problem')
    
    file_name = 'h-convergence-convection.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(test_dir, file_name), dpi = 300)
    plt.close(fig)
