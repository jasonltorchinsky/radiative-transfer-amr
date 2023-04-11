import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import os, sys

sys.path.append('../')
from test_cases import get_test_funcs

sys.path.append('../../src')
from dg.mesh import Mesh, get_hasnt_th
from dg.mesh.utils import plot_mesh, plot_mesh_bdry

from dg.projection import Projection, intg_th
from dg.projection.utils import plot_projection, plot_angular_dists

from utils import print_msg


def test_2(dir_name = 'test_mesh'):
    """
    Angular integration of a projection.
    """
    
    test_dir = os.path.join(dir_name, 'test_2')
    os.makedirs(dir_name, exist_ok = True)
    
    mesh_dir  = os.path.join(test_dir, 'mesh')
    os.makedirs(mesh_dir, exist_ok = True)
    
    proj_dir  = os.path.join(test_dir, 'proj')
    os.makedirs(proj_dir, exist_ok = True)

    intg_dir = os.path.join(test_dir, 'intg')
    os.makedirs(intg_dir, exist_ok = True)

    # Select test problem
    func_num = 2
    [test_func_2d, test_func_3d] = get_test_funcs(func_num)
    
    # Create the original 2-D mesh
    [Lx, Ly] = [3., 2.]
    ndofs    = [2, 2, 6]
    pbcs     = [False, False]
    
    mesh = Mesh([Lx, Ly], pbcs, ndofs, has_th = True)
    mesh.ref_mesh(kind = 'ang')
    mesh.ref_mesh(kind = 'ang')
    mesh.ref_mesh(kind = 'spt')
    
    proj = Projection(mesh, test_func_3d)
    
    file_name = os.path.join(mesh_dir, 'mesh_3d_init.png')
    plot_mesh(mesh, ax = None, file_name = file_name,
              label_cells = False, plot_dim = 3)

    angles = [0, np.pi / 2., np.pi, 3. * np.pi / 2.]
    file_name = os.path.join(proj_dir, 'projs_init.png')
    plot_projection(mesh, proj, file_name = file_name,
                    angles = angles)
    
    ntrial = 3
    ndofs = np.zeros([ntrial])
    errs = np.zeros([ntrial])
    for trial in range(0, ntrial):
        perf_trial_0 = perf_counter()
        print_msg('[Trial {}] Starting...'.format(trial))
        
        mesh_2d = get_hasnt_th(mesh)
        
        proj_intg_th = intg_th(mesh, proj)
        vec_intg_th  = proj_intg_th.to_vector()

        file_name = os.path.join(intg_dir, 'intg_th_num_{}.png'.format(trial))
        plot_projection(mesh_2d, proj_intg_th, file_name = file_name)
        
        proj_intg_th_anl = Projection(mesh_2d, test_func_2d)
        vec_intg_th_anl = proj_intg_th_anl.to_vector()

        file_name = os.path.join(intg_dir, 'intg_th_anl_{}.png'.format(trial))
        plot_projection(mesh_2d, proj_intg_th_anl, file_name = file_name)
        
        ndofs[trial] = np.size(vec_intg_th)
        errs[trial]  = np.amax(np.abs(vec_intg_th - vec_intg_th_anl))
        
        file_name = os.path.join(mesh_dir, 'mesh_3d_{}.png'.format(trial))
        plot_mesh(mesh, ax = None, file_name = file_name,
                  label_cells = False, plot_dim = 3)
        
        file_name = os.path.join(proj_dir, 'projs_{}.png'.format(trial))
        plot_projection(mesh, proj, file_name = file_name,
                        angles = angles)

        file_name = os.path.join(proj_dir, 'angular_dists_{}.png'.format(trial))
        plot_angular_dists(mesh, proj, file_name = file_name)
        
        mesh.ref_mesh(kind = 'all')
        proj = Projection(mesh, test_func_3d)

        perf_trial_f    = perf_counter()
        perf_trial_diff = perf_trial_f - perf_trial_0
        msg = (
            '[Trial {}] Trial completed! '.format(trial) +
            'Time Elapsed: {:08.3f} [s]\n'.format(perf_trial_diff)
        )
        print_msg(msg)
        
    file_name = os.path.join(mesh_dir, 'mesh_2d_bdry.png')
    plot_mesh_bdry(mesh, file_name = file_name,
                   label_cells = False, plot_dim = 2)
    
    # Plot errors
    fig, ax = plt.subplots()
    
    ax.plot(ndofs, errs,
            color     = 'k',
            linestyle = '-')

    ax.set_xscale('log', base = 2)
    ax.set_yscale('log', base = 2)

    max_err = max(errs)
    min_err = min(errs)
    ymin = 2**(np.floor(np.log2(min_err)))
    ymax = 2**(np.ceil(np.log2(max_err)))
    ax.set_ylim([ymin, ymax])
    
    ax.set_xlabel('Total Degrees of Freedom')
    ax.set_ylabel('L$^{\infty}$ Error')
    
    title_str = 'Angular Integration Error'
    ax.set_title(title_str)
    
    file_name = 'intg_th_err.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(test_dir, file_name), dpi = 300)
    plt.close(fig)
