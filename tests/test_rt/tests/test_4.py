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
from dg.matrix import get_intr_mask, split_matrix, merge_vectors
from dg.projection import push_forward, to_projection
from dg.projection.utils import plot_projection
import dg.quadrature as qd
from rt import calc_intr_conv_matrix, calc_bdry_conv_matrix

from utils import print_msg

def test_4(dir_name = 'test_rt'):
    """
    Creates various visualizations of the convection matrix and solves a 
    manufactured problem.
    """
    
    test_dir = os.path.join(dir_name, 'test_4')
    os.makedirs(test_dir, exist_ok = True)

    # Set the refinement type: 'sin' - single column
    #                        : 'uni' - uniform
    #                        : 'amr' - adaptive
    ref_type = 'uni'
    ntrial   = 4
    
    # Get the base mesh, manufactured solution
    [Lx, Ly]                   = [3., 2.]
    pbcs                       = [False, False]
    [ndof_x, ndof_y, ndof_th]  = [2, 2, 2]
    has_th                     = True
    mesh = gen_mesh(Ls     = [Lx, Ly],
                    pbcs   = pbcs,
                    ndofs  = [ndof_x, ndof_y, ndof_th],
                    has_th = has_th)
    
    [anl_sol, _, _, _, f] = get_cons_soln(prob_name = 'conv',
                                          sol_num   = 2)
    
    # Solve simplified problem over several trials
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
        [M_intr, M_bdry] = split_matrix(mesh, M_conv, intr_mask)
        
        apr_sol_vec_intr = spsolve(M_intr, f_vec_intr - M_bdry @ bcs_vec)
        
        perf_soln_f    = perf_counter()
        perf_soln_diff = perf_soln_f - perf_soln_0
        msg = (
            '[Trial {}] Manufactured problem solved! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]'.format(perf_soln_diff)
        )
        print_msg(msg)

        # Plot the difference in solutions
        diff_vec_intr = apr_sol_vec_intr - anl_sol_vec_intr
        zero_bcs_vec  = 0. * bcs_vec
        diff_vec      = merge_vectors(diff_vec_intr, zero_bcs_vec, intr_mask)
        diff_proj     = to_projection(mesh, diff_vec)
        
        file_name = os.path.join(trial_dir, 'diff.png')
        angles = [0, np.pi/3, 2 * np.pi / 3, np.pi,
                  4 * np.pi / 3, 5 * np.pi / 3]
        plot_projection(diff_proj, file_name = file_name, angles = angles)

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

        #file_name = 'conv_matrix.csv'
        #np.savetxt(os.path.join(trial_dir, file_name), M_conv.toarray(), delimiter = ',')
        
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
        if ref_type == 'sin':
            ## Refine a given column spatially
            col_keys = sorted(mesh.cols.keys())
            mesh.ref_col(col_keys[-4], kind = 'spt')
        elif ref_type == 'uni':
            ## Refine the mesh uniformly spatially
            mesh.ref_mesh(kind = 'spt')
        elif ref_type == 'amr':
            ## Refine the column spatially with the biggest "error"
            max_err = 0
            col_errs = {}
            diff_proj_col_items = sorted(diff_proj.cols.items())
            for col_key, col in diff_proj_col_items:
                if col.is_lf:
                    col_err = 0
                    cell_items = sorted(col.cells.items())
                    for cell_key, cell in cell_items:
                        if cell.is_lf:
                            cell_err = np.amax(np.abs(cell.vals))
                            col_err = max(col_err, cell_err)
                            
                    col_errs[col_key] = col_err
                    max_err = max(max_err, col_err)
                    
            col_keys = sorted(mesh.cols.keys())
            for col_key in col_keys:
                if col_key in mesh.cols.keys():
                    col = mesh.cols[col_key]
                    if col.is_lf:
                        col_err = col_errs[col_key]
                        if col_err > 0.9 * max_err:
                            mesh.ref_col(col_key, kind = 'spt')
            
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
    title_str = '{} Spatial $h$-Refinement Convergence Rate - Convection Problem'.format(ref_str)
    ax.set_title(title_str)
    
    file_name = 'h-convergence-convection.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(test_dir, file_name), dpi = 300)
    plt.close(fig)
