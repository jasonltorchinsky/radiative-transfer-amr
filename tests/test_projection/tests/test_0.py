import numpy as np
import matplotlib.pyplot as plt
import os, sys

from .test_funcs import func_2D_1 as test_func

sys.path.append('../../src')
import dg.mesh as ji_mesh
import dg.mesh.utils

import dg.projection
import dg.projection.utils


def test_0(dir_name = 'test_mesh'):
    """
    2D projection creation and plotting.
    """
    
    dir_name = os.path.join(dir_name, 'test_0')
    os.makedirs(dir_name, exist_ok = True)
    
    mesh_dir  = os.path.join(dir_name, 'mesh')
    os.makedirs(mesh_dir, exist_ok = True)
    
    proj_dir  = os.path.join(dir_name, 'proj')
    os.makedirs(proj_dir, exist_ok = True)
    
    # Create the original 2-D mesh
    [Lx, Ly] = [3., 2.]
    ndofs    = [4, 3, 4]
    pbcs     = [False, True]
    
    mesh = ji_mesh.Mesh([Lx, Ly], pbcs, ndofs, has_th = False)
    proj = dg.projection.Projection(mesh, test_func)
    
    # Refine the mesh some so we have a more interesting plot
    nrefs = 0
    
    file_name = os.path.join(mesh_dir, 'mesh_2d_{}.png'.format(nrefs))
    ji_mesh.utils.plot_mesh(mesh, ax = None, file_name = file_name,
                            label_cells = (nrefs <= 3), plot_dim = 2)
    
    file_name = os.path.join(proj_dir, 'proj_{}.png'.format(nrefs))
    dg.projection.utils.plot_projection(proj, file_name = file_name)
    
    nspt_ref = 4
    for ref in range(0, nspt_ref):
        
        col_keys = sorted(list(mesh.cols.keys()))
        mesh.ref_col(col_keys[-1], kind = 'spt')
        
        #mesh.ref_mesh(kind = 'spt')
        proj = dg.projection.Projection(mesh, test_func)
        nrefs += 1
        
        file_name = os.path.join(mesh_dir, 'mesh_2d_{}.png'.format(nrefs))
        ji_mesh.utils.plot_mesh(mesh, file_name = file_name)
        
        file_name = os.path.join(proj_dir, 'proj_{}.png'.format(nrefs))
        dg.projection.utils.plot_projection(proj, file_name = file_name)

    file_name = os.path.join(mesh_dir, 'mesh_2d_bdry.png')
    ji_mesh.utils.plot_mesh_bdry(mesh, file_name = file_name,
                                 label_cells = False, plot_dim = 2)
    
