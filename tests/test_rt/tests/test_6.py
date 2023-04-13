import copy
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import os, sys

from .gen_mesh import gen_mesh

sys.path.append('../../tests')
from test_cases import get_cons_prob

sys.path.append('../../src')
from dg.mesh.utils import plot_mesh
from dg.projection import Projection
from dg.projection.utils import plot_projection, plot_angular_dists
from rt import rtdg
from amr import col_jump_err

from utils import print_msg


def test_6(dir_name = 'test_rt'):
    """
    Solves a test problem of the full system.
    """
    
    test_dir = os.path.join(dir_name, 'test_6')
    os.makedirs(test_dir, exist_ok = True)

    # Set the refinement type: 'sin' - single column
    #                        : 'uni' - uniform
    #                        : 'amr' - adaptive
    ref_type = 'amr'
    ntrial   = 3
    
    # Get the base mesh, test_problem
    [Lx, Ly]                   = [2., 3.]
    pbcs                       = [True, False]
    [ndof_x, ndof_y, ndof_th]  = [1, 1, 1]
    has_th                     = True
    mesh = gen_mesh(Ls     = [Lx, Ly],
                    pbcs   = pbcs,
                    ndofs  = [ndof_x, ndof_y, ndof_th],
                    has_th = has_th)
    mesh.ref_mesh(kind = 'spt')
    
    [kappa, sigma, Phi, [bcs, dirac], f] = get_test_prob(prob_num = 2,
                                                         mesh = mesh)
    
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
        
        u_proj = rtdg(mesh, kappa, sigma, Phi, [bcs, dirac], f)
        
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

        file_name = os.path.join(trial_dir, 'soln_slices.png')
        plot_angular_dists(mesh, u_proj, file_name = file_name)
        
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
            'Time Elapsed: {:08.3f} [s]\n'.format(perf_trial_diff)
        )
        print_msg(msg)
        
