import numpy as np
import matplotlib.pyplot as plt
import os, sys

from .test_funcs import func_2D_1 as test_func_2D
from .test_funcs import func_3D_1 as test_func_3D

sys.path.append('../../src')
from dg.mesh import Mesh, get_hasnt_th
from dg.mesh.utils import plot_mesh, plot_mesh_bdry

from dg.projection import Projection, intg_th
from dg.projection.utils import plot_projection


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
    
    # Create the original 2-D mesh
    [Lx, Ly] = [3., 2.]
    ndofs    = [4, 3, 4]
    pbcs     = [False, False]
    
    mesh = Mesh([Lx, Ly], pbcs, ndofs, has_th = True)
    mesh.ref_mesh(kind = 'ang')
    mesh.ref_mesh(kind = 'ang')
    mesh.ref_mesh(kind = 'spt')
    
    proj = Projection(mesh, test_func_3D)
    
    file_name = os.path.join(mesh_dir, 'mesh_3d_{}.png'.format(0))
    plot_mesh(mesh, ax = None, file_name = file_name,
              label_cells = False, plot_dim = 3)

    angles = [0, np.pi / 2., np.pi, 3. * np.pi / 2.]
    file_name = os.path.join(proj_dir, 'projs_{}.png'.format(0))
    plot_projection(mesh, proj, file_name = file_name,
                    angles = angles)
    
    nref = 3
    ndofs = np.zeros([nref])
    errs = np.zeros([nref])
    for ref in range(0, nref):
        proj_intg_th = intg_th(mesh, proj)
        vec_intg_th  = proj_intg_th.to_vector()

        mesh_2d = get_hasnt_th(mesh)
        proj_intg_th_anl = Projection(mesh_2d, test_func_2D)
        vec_intg_th_anl = proj_intg_th_anl.to_vector()
        
        ndofs[ref] = np.size(vec_intg_th)
        errs[ref] = np.amax(np.abs(vec_intg_th - vec_intg_th_anl))
        
        file_name = os.path.join(mesh_dir, 'mesh_3d_{}.png'.format(ref + 1))
        plot_mesh(mesh, ax = None, file_name = file_name,
                  label_cells = False, plot_dim = 3)
        
        file_name = os.path.join(proj_dir, 'projs_{}.png'.format(ref + 1))
        plot_projection(mesh, proj, file_name = file_name,
                        angles = angles)
        
        mesh.ref_mesh(kind = 'all')
        proj = Projection(mesh, test_func_3D)
        
    file_name = os.path.join(mesh_dir, 'mesh_2d_bdry.png')
    plot_mesh_bdry(mesh, file_name = file_name,
                   label_cells = False, plot_dim = 2)
    
    # Plot errors
    fig, ax = plt.subplots()
    
    ax.plot(ndofs, errs,
            label     = 'L$^{\infty}$ Error',
            color     = 'k',
            linestyle = '-')

    ax.legend()

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
