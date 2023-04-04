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
from dg.projection import Projection, push_forward, to_projection
from dg.projection.utils import plot_projection
import dg.quadrature as qd
from rt import calc_mass_matrix, calc_scat_matrix, \
    calc_intr_conv_matrix, calc_bdry_conv_matrix
from amr import col_jump_err

from utils import print_msg


def test_1(dir_name = 'test_amr'):
    """
    Tests the jump error calculation.
    """
    
    test_dir = os.path.join(dir_name, 'test_1')
    os.makedirs(test_dir, exist_ok = True)

    # Set the refinement type: 'sin' - single column
    #                        : 'uni' - uniform
    #                        : 'amr' - adaptive
    ref_type = 'uni'
    ntrial   = 3
    
    # Get the base mesh, test_problem
    [Lx, Ly]                   = [3., 2.]
    pbcs                       = [True, False]
    [ndof_x, ndof_y, ndof_th]  = [2, 2, 2]
    has_th                     = True
    mesh = gen_mesh(Ls     = [Lx, Ly],
                    pbcs   = pbcs,
                    ndofs  = [ndof_x, ndof_y, ndof_th],
                    has_th = has_th)
    
    [anl_sol, kappa, sigma, Phi, f, anl_sol_intg_th] = get_cons_soln(prob_name = 'comp',
                                                                     sol_num   = 0)
    
    # Solve simplified problem over several trials
    trial_sol_errs  = np.zeros([ntrial])
    trial_intg_errs = np.zeros([ntrial])
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


        f_vec       = get_forcing_vec(mesh, f)
        anl_sol_vec = get_projection_vec(mesh, anl_sol)
        
        intr_mask        = get_intr_mask(mesh)
        bdry_mask        = np.invert(intr_mask)
        f_vec_intr       = f_vec[intr_mask]
        anl_sol_vec_intr = anl_sol_vec[intr_mask]
        bcs_vec          = anl_sol_vec[bdry_mask]
        
        M = (M_bdry_conv - M_intr_conv) + M_mass - M_scat
        [M_intr, M_bdry] = split_matrix(mesh, M, intr_mask)
        
        u_vec_intr = spsolve(M_intr, f_vec_intr - M_bdry @ bcs_vec)
        u_vec      = merge_vectors(u_vec_intr, bcs_vec, intr_mask)
        u_proj     = to_projection(mesh, u_vec)
        
        perf_cons_f    = perf_counter()
        perf_cons_diff = perf_cons_f - perf_cons_0
        msg = (
            '[Trial {}] Test problem solved! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]'.format(perf_cons_diff)
        )
        print_msg(msg)

        # Calculate the maximum error of the solution
        anl_sol_proj = Projection(mesh, anl_sol)
        anl_sol_vec  = anl_sol_proj.to_vector()
        diff_vec     = np.abs(u_vec - anl_sol_vec)
        max_err      = np.amax(diff_vec)
        msg = (
            '[Trial {}] Max solution error: {} '.format(trial, max_err)
        )
        print_msg(msg)

        # Calculate the error of angular integration
        max_err      = 0.
        col_jump_errs = col_jump_err(mesh, u_proj)
        print(col_jump_errs)

        #msg = (
        #    '[Trial {}] Max integration error: {} '.format(trial, max_err)
        #)
        #print_msg(msg)
        
        
        # Refine the mesh for the next trial
        if ref_type == 'sin':
            ## Refine a given column
            col_keys = sorted(mesh.cols.keys())
            mesh.ref_col(col_keys[-4], kind = 'all')
        elif ref_type == 'uni':
            ## Refine the mesh uniformly
            mesh.ref_mesh(kind = 'spt')
            
        perf_trial_f    = perf_counter()
        perf_trial_diff = perf_trial_f - perf_trial_0
        msg = (
            '[Trial {}] Trial completed! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]\n'.format(perf_trial_diff)
        )
        print_msg(msg)
        
