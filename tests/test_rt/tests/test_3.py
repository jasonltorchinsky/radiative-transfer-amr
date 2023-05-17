import copy
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import os, sys

from .gen_mesh import gen_mesh

sys.path.append('../../tests')
from test_cases import get_test_prob

sys.path.append('../../src')
from dg.mesh import get_hasnt_th
from dg.mesh.utils import plot_mesh
from dg.projection import Projection, intg_th
from dg.projection.utils import plot_projection, plot_angular_dists
from rt import rtdg
from amr import cell_jump_err, col_jump_err, ref_by_ind

from utils import print_msg


def test_3(dir_name = 'test_rt'):
    """
    Solves test problems, using the built-in RT solver.
    """
    
    test_dir = os.path.join(dir_name, 'test_3')
    os.makedirs(test_dir, exist_ok = True)

    # Test parameters:
    # Problem Number
    prob_num  = 1
    # Refinement Type: 'sin'gle column, 'uni'form, 'a'daptive 'm'esh 'r'efinement
    ref_type = 'uni'
    # Refinement Kind: 's'pa't'ia'l', 'ang'ular, 'all'
    ref_kind = 'ang'
    # Refinement Form: 'h', 'p'
    ref_form = 'h'
    # AMR Refinement Tolerance
    tol = 0.9
    # Maximum number of DOFs
    max_ndof = 2**13
    # Maximum number of trials
    max_ntrial = 6

    # Test Output Parameters
    do_plot_mesh        = True
    do_plot_coeff_funcs = False
    do_plot_uh          = True
    
    for ref_type in ['amr']:
        for ref_kind in ['spt']:
            for ref_form in ['hp']:
                msg = (
                    'Staring combination {}, {}-{}.\n'.format(ref_kind,
                                                              ref_form,
                                                              ref_type)
                    )
                print_msg(msg)
                
                dir_name = '{}-{}-{}'.format(ref_kind, ref_form, ref_type)
                combo_dir = os.path.join(test_dir, dir_name)
                
                # Get the base mesh, test problem
                [Lx, Ly]                   = [2., 3.]
                pbcs                       = [True, False]
                [ndof_x, ndof_y, ndof_th]  = [2, 2, 6]
                has_th                     = True
                mesh = gen_mesh(Ls     = [Lx, Ly],
                                pbcs   = pbcs,
                                ndofs  = [ndof_x, ndof_y, ndof_th],
                                has_th = has_th)
                
                [kappa, sigma, Phi, [bcs, dirac], f] = get_test_prob(
                    prob_num = prob_num,
                    mesh     = mesh)
                
                # Solve the manufactured problem over several trials
                ref_ndofs = []
                
                ndof = 0
                trial = 0
                while (ndof < max_ndof) and (trial <= max_ntrial):
                    perf_trial_0 = perf_counter()
                    print_msg('[Trial {}] Starting...'.format(trial))
                    
                    # Set up output directories
                    trial_dir = os.path.join(combo_dir, 'trial_{}'.format(trial))
                    os.makedirs(trial_dir, exist_ok = True)
                    
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
                    
                    uh_vec = uh_proj.to_vector()
                    ndof = np.size(uh_vec)
                    ref_ndofs += [ndof]

                    msg = (
                        '[Trial {}] Number of DOFs: {} '.format(trial, ndof) +
                        'of {}.'.format(max_ndof)
                    )
                    print_msg(msg)
                    
                    mesh_2d = get_hasnt_th(mesh)
                    
                    if do_plot_mesh:
                        file_name = os.path.join(trial_dir, 'mesh_3d.png')
                        plot_mesh(mesh      = mesh,
                                  file_name = file_name,
                                  plot_dim  = 3)
                        file_name = os.path.join(trial_dir, 'mesh_2d.png')
                        plot_mesh(mesh        = mesh,
                                  file_name   = file_name,
                                  plot_dim    = 2,
                                  label_cells = (trial <= 3))
                        
                    if do_plot_coeff_funcs:
                        # Plot the coefficient functions
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
                        
                    if do_plot_uh:
                        #file_name = os.path.join(trial_dir, 'uh_proj.png')
                        #angles = [0, np.pi/3, 2 * np.pi / 3, np.pi,
                        #          4 * np.pi / 3, 5 * np.pi / 3]
                        #plot_projection(mesh, uh_proj, file_name = file_name, angles = angles)
                        
                        file_name = os.path.join(trial_dir, 'uh_ang_dist.png')
                        plot_angular_dists(mesh, uh_proj, file_name = file_name)
                        
                        mean_uh = intg_th(mesh, uh_proj)
                        file_name = os.path.join(trial_dir, 'uh_mean.png')
                        plot_projection(mesh_2d, mean_uh, file_name = file_name)
                        
                        
                    # Refine the mesh for the next trial
                    if ref_type == 'sin':
                        ## Refine a given column
                        col_keys = sorted(mesh.cols.keys())
                        mesh.ref_col(col_keys[-1], kind = ref_kind, form = ref_form)
                    elif ref_type == 'uni':
                        ## Refine the mesh uniformly
                        mesh.ref_mesh(kind = ref_kind, form = ref_form)
                    elif ref_type == 'amr':
                        if ref_kind in ['ang', 'all']:
                            cell_jump_err_ind = cell_jump_err(mesh, uh_proj)
                        if ref_kind in ['spt', 'all']:
                            col_jump_err_ind = col_jump_err(mesh, uh_proj)
                            
                        if ref_kind in ['ang', 'all']:
                            mesh = ref_by_ind(mesh, cell_jump_err_ind,
                                          ref_ratio = tol, form = ref_form)
                        if ref_kind in ['spt', 'all']:
                            mesh = ref_by_ind(mesh, col_jump_err_ind,
                                          ref_ratio = tol, form = ref_form)
                        
                        
                        
                        
                    perf_trial_f    = perf_counter()
                    perf_trial_diff = perf_trial_f - perf_trial_0
                    msg = (
                        '[Trial {}] Trial completed! '.format(trial) +
                        'Time Elapsed: {:08.3f} [s]\n'.format(perf_trial_diff)
                    )
                    print_msg(msg)
                    
                    trial += 1
