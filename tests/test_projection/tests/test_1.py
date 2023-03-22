import numpy as np
import matplotlib.pyplot as plt
import os, sys

from .test_funcs import func_3D_1 as test_func

sys.path.append('../../src')
import dg.mesh as ji_mesh
import dg.mesh.utils

import dg.projection
import dg.projection.utils


def test_1(dir_name = 'test_mesh'):
    """
    3D projection creation and plotting.
    """
    
    dir_name = os.path.join(dir_name, 'test_1')
    os.makedirs(dir_name, exist_ok = True)
    
    mesh_dir  = os.path.join(dir_name, 'mesh')
    os.makedirs(mesh_dir, exist_ok = True)
    
    proj_dir  = os.path.join(dir_name, 'proj')
    os.makedirs(proj_dir, exist_ok = True)
    
    # Create the original 2-D mesh
    [Lx, Ly] = [3., 2.]
    ndofs    = [4, 3, 4]
    pbcs     = [False, False]
    
    mesh = ji_mesh.Mesh([Lx, Ly], pbcs, ndofs, has_th = True)
    proj = dg.projection.Projection(mesh, test_func)
    
    # Refine the mesh some so we have a more interesting plot
    angles = [0, 2 * np.pi / 3, 4 * np.pi /3, 2 * np.pi]
    nrefs = 0
    
    file_name = os.path.join(mesh_dir, 'mesh_3d_{}.png'.format(nrefs))
    ji_mesh.utils.plot_mesh(mesh, ax = None, file_name = file_name,
                            label_cells = False, plot_dim = 3)
    
    file_name = os.path.join(proj_dir, 'projs_{}.png'.format(nrefs))
    dg.projection.utils.plot_projection(proj, file_name = file_name,
                                        angles = angles)
    
    nref = 6
    for ref in range(0, nref):
        
        col_keys = sorted(list(mesh.cols.keys()))
        col_key = col_keys[-1]
        col = mesh.cols[col_key]
        cell_keys = sorted(list(col.cells.keys()))
        cell_key = cell_keys[-1]
        mesh.ref_cell(col_key, cell_key)
        mesh.ref_col(col_key, kind = 'spt')
        
        #mesh.ref_mesh(kind = 'all')
        proj = dg.projection.Projection(mesh, test_func)
        nrefs += 1
        
        file_name = os.path.join(mesh_dir, 'mesh_3d_{}.png'.format(nrefs))
        ji_mesh.utils.plot_mesh(mesh, ax = None, file_name = file_name,
                                label_cells = False, plot_dim = 3)
        
        file_name = os.path.join(proj_dir, 'projs_{}.png'.format(nrefs))
        dg.projection.utils.plot_projection(proj, file_name = file_name,
                                            angles = angles)

    file_name = os.path.join(mesh_dir, 'mesh_2d_bdry.png')
    ji_mesh.utils.plot_mesh_bdry(mesh, file_name = file_name,
                                 label_cells = False, plot_dim = 2)
    
