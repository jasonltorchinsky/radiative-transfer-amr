import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append('../../src')
import dg.mesh as ji_mesh
import dg.mesh.utils

def test_3(dir_name = 'test_mesh'):
    """
    Mixed non-uniform angular/spatial refinement, and neighbors for cells.
    """

    dir_name = os.path.join(dir_name, 'test_3')
    os.makedirs(dir_name, exist_ok = True)

    mesh_dir  = os.path.join(dir_name, 'mesh')
    os.makedirs(mesh_dir, exist_ok = True)
    
    nhbrs_dir = os.path.join(dir_name, 'nhbrs')
    os.makedirs(nhbrs_dir, exist_ok = True)

    # Create the original 2-D mesh
    [Lx, Ly] = [3, 2]
    pbcs     = [True, False]

    mesh = ji_mesh.Mesh([Lx, Ly], pbcs, has_th = True)
    nang_refs = 2
    for ref in range(0, nang_refs):
        mesh.ref_mesh(kind = 'ang')
    
    # Refine the mesh some so we have a more interesting plot
    nrefs = 0
    
    file_name = os.path.join(mesh_dir, 'mesh_2d_{}.png'.format(nrefs))
    ji_mesh.utils.plot_mesh(mesh, ax = None, file_name = file_name,
                            label_cells = (nrefs <= 3), plot_dim = 2)
    
    nuni_ref = 1
    for ref in range(0, nuni_ref):
        mesh.ref_mesh(kind = 'spt')
        nrefs += 1
        
        file_name = os.path.join(mesh_dir, 'mesh_2d_{}.png'.format(nrefs))
        ji_mesh.utils.plot_mesh(mesh, ax = None, file_name = file_name,
                                label_cells = (nrefs <= 3), plot_dim = 2)

    ncol_ref = 2
    for ref in range(0, ncol_ref):
        col_keys = sorted(list(mesh.cols.keys()))
        mesh.ref_col(col_keys[-1], kind = 'spt')
        nrefs += 1
        
        file_name = os.path.join(mesh_dir, 'mesh_2d_{}.png'.format(nrefs))
        ji_mesh.utils.plot_mesh(mesh, ax = None, file_name = file_name,
                                label_cells = (nrefs <= 3), plot_dim = 2)

    ncell_ref = 2
    for ref in range(0, ncell_ref):
        col_keys = sorted(list(mesh.cols.keys()))
        cell_keys = sorted(list(mesh.cols[col_keys[-1]].cells.keys()))
        mesh.ref_cell(col_keys[-1], cell_keys[-1])
        nrefs += 1
        
        file_name = os.path.join(mesh_dir, 'mesh_2d_{}.png'.format(nrefs))
        ji_mesh.utils.plot_mesh(mesh, ax = None, file_name = file_name,
                                label_cells = (nrefs <= 3), plot_dim = 2)
        
        file_name = os.path.join(mesh_dir, 'mesh_3d_{}.png'.format(nrefs))
        ji_mesh.utils.plot_mesh(mesh, ax = None, file_name = file_name,
                                label_cells = (nrefs <= 3), plot_dim = 3,
                                plot_style = 'flat')

        file_name = os.path.join(mesh_dir, 'mesh_3d`_flat_{}.png'.format(nrefs))
        ji_mesh.utils.plot_mesh(mesh, ax = None, file_name = file_name,
                                label_cells = (nrefs <= 3), plot_dim = 3,
                                plot_style = 'box')

    file_name = os.path.join(mesh_dir, 'mesh_2d_bdry.png')
    ji_mesh.utils.plot_mesh_bdry(mesh, file_name = file_name,
                                 label_cells = (nrefs <= 3), plot_dim = 2)

    file_name = os.path.join(mesh_dir, 'mesh_3d_bdry.png')
    ji_mesh.utils.plot_mesh_bdry(mesh, file_name = file_name,
                                 label_cells = (nrefs <= 3), plot_dim = 3)
        
    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            for cell_key, cell in sorted(col.cells.items()):
                file_str  = 'cell_{}_{}_nhbrs.png'.format(col_key, cell_key)
                #file_name = os.path.join(nhbrs_dir, file_str)
                #ji_mesh.utils.plot_cell_nhbrs(mesh, col, cell, file_name = file_name)
