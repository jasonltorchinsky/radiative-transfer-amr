import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append('../../src')
from dg.mesh import ji_mesh, tools

def test_2(dir_name = 'test_mesh'):
    """
    Creates a simple 3-D mesh and tests the find-neighbors function for cells.
    """

    dir_name = os.path.join(dir_name, 'test_2')
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
    fig, ax = plt.subplots()
    file_name = os.path.join(mesh_dir, 'mesh_2d_{}.png'.format(nrefs))
    tools.plot_mesh(mesh, ax = ax, file_name = file_name,
                    label_cells = True, plot_dim = 2)
    
    nuni_ref = 1
    for ref in range(0, nuni_ref):
        mesh.ref_mesh(kind = 'spt')
        nrefs += 1
        fig, ax = plt.subplots()
        file_name = os.path.join(mesh_dir, 'mesh_2d_{}.png'.format(nrefs))
        tools.plot_mesh(mesh, ax = ax, file_name = file_name,
                    label_cells = True, plot_dim = 2)

    ncol_ref = 2
    for ref in range(0, ncol_ref):
        col_keys = sorted(list(mesh.cols.keys()))
        mesh.ref_col(mesh.cols[col_keys[-1]], kind = 'all')
        nrefs += 1
        file_name = os.path.join(mesh_dir, 'mesh_2d_{}.png'.format(nrefs))
        tools.plot_mesh(mesh, ax = None, file_name = file_name,
                        label_cells = True, plot_dim = 2)
        
        file_name = os.path.join(mesh_dir, 'mesh_3d_{}.png'.format(nrefs))
        tools.plot_mesh(mesh, ax = None, file_name = file_name,
                        label_cells = False, plot_dim = 3)

    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            for cell_key, cell in sorted(col.cells.items()):
                file_str  = 'cell_{}_{}_nhbrs.png'.format(col_key, cell_key)
                file_name = os.path.join(nhbrs_dir, file_str)
                tools.plot_cell_nhbrs(mesh, col, cell, file_name = file_name)
