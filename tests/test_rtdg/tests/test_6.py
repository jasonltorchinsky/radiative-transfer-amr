import copy
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import os, sys

from .gen_mesh           import gen_mesh
from .get_test_prob      import get_test_prob

sys.path.append('../../src')
from dg.mesh.utils import plot_mesh
from dg.projection import Projection
from dg.projection.utils import plot_projection
from rt import rtdg

from utils import print_msg


def test_6(dir_name = 'test_rtdg'):
    """
    Solves a test problem of the full system.
    """
    
    test_dir = os.path.join(dir_name, 'test_6')
    os.makedirs(test_dir, exist_ok = True)

    # Set the refinement type: 'sin' - single column
    #                        : 'uni' - uniform
    #                        : 'amr' - adaptive
    ref_type = 'uni'
    ntrial   = 3
    
    # Get the base mesh, test_problem
    [Lx, Ly]                   = [3., 2.]
    pbcs                       = [True, False]
    [ndof_x, ndof_y, ndof_th]  = [4, 4, 4]
    has_th                     = True
    mesh = gen_mesh(Ls     = [Lx, Ly],
                    pbcs   = pbcs,
                    ndofs  = [ndof_x, ndof_y, ndof_th],
                    has_th = has_th)
    
    [kappa, sigma, Phi, bcs, f] = get_test_prob(prob_num = 0)
    
    # Solve simplified problem over several trials
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
        mesh_2d = copy.deepcopy(mesh)
        mesh_2d.has_th = False
        
        kappa_proj = Projection(mesh_2d, kappa)
        file_name = os.path.join(trial_dir, 'kappa.png')
        plot_projection(kappa_proj, file_name = file_name)
        
        sigma_proj = Projection(mesh_2d, sigma)
        file_name = os.path.join(trial_dir, 'sigma.png')
        plot_projection(sigma_proj, file_name = file_name)
        
        # Construct solve the test problem
        perf_cons_0 = perf_counter()
        print_msg('[Trial {}] Solving the test problem...'.format(trial))
        
        u_proj = rtdg(mesh, kappa, sigma, Phi, bcs, f)
        
        perf_cons_f    = perf_counter()
        perf_cons_diff = perf_cons_f - perf_cons_0
        msg = (
            '[Trial {}] Test problem solved! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]'.format(perf_cons_diff)
        )
        print_msg(msg)

        # Plot the solution
        file_name = os.path.join(trial_dir, 'soln.png')
        angles = np.linspace(0, 1.75, 8) * np.pi
        plot_projection(u_proj, file_name = file_name, angles = angles)
        
        # Refine the mesh for the next trial
        if ref_type == 'sin':
            ## Refine a given column
            col_keys = sorted(mesh.cols.keys())
            mesh.ref_col(col_keys[-4], kind = 'all')
        elif ref_type == 'uni':
            ## Refine the mesh uniformly
            mesh.ref_mesh(kind = 'ang')
            
        perf_trial_f    = perf_counter()
        perf_trial_diff = perf_trial_f - perf_trial_0
        msg = (
            '[Trial {}] Trial completed! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]'.format(perf_trial_diff)
        )
        print_msg(msg)
        