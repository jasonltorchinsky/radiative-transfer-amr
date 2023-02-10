import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append('../../src')
from dg.mesh import ji_mesh, tools

def test_1(dir_name = 'test_mesh'):
    """
    Creates a simple 3-D mesh and tests the find-neighbors function.
    """

    dir_name = os.path.join(dir_name, 'test_1')
    os.makedirs(dir_name, exist_ok = True)

    # Create the original 2-D mesh
    [Lx, Ly] = [3, 2]
    pbcs     = [True, False]

    mesh = ji_mesh.Mesh([Lx, Ly], pbcs, has_th = True)
    ncol_refs = 3
    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            for ref in range(0, ncol_refs):
                col.ref_col()
            
    
    # Refine the mesh some so we have a more interesting plot
    nrefs = 0
    fig, ax = plt.subplots()
    file_name = os.path.join(dir_name, 'mesh_2d_{}.png'.format(nrefs))
    tools.plot_mesh(mesh, ax = ax, file_name = file_name,
                    label_cells = True, plot_dim = 2)
    
    nuni_ref = 2
    for ref in range(0, nuni_ref):
        mesh.ref_mesh()
        nrefs += 1
        fig, ax = plt.subplots()
        file_name = os.path.join(dir_name, 'mesh_2d_{}.png'.format(nrefs))
        tools.plot_mesh(mesh, ax = ax, file_name = file_name,
                    label_cells = True, plot_dim = 2)

    ncol_ref = 5
    for ref in range(0, ncol_ref):
        col_keys = sorted(list(mesh.cols.keys()))
        mesh.ref_col(mesh.cols[col_keys[-1]])
        nrefs += 1
        fig, ax = plt.subplots()
        file_name = os.path.join(dir_name, 'mesh_2d_{}.png'.format(nrefs))
        tools.plot_mesh(mesh, ax = ax, file_name = file_name,
                        label_cells = False, plot_dim = 2)

    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            file_name = os.path.join(dir_name, 'col_{}_nhbrs_2D.png'.format(col_key))
            tools.plot_col_nhbrs(mesh, col, file_name = file_name, plot_dim = 2)
            
            file_name = os.path.join(dir_name, 'col_{}_nhbrs_3D.png'.format(col_key))
            tools.plot_col_nhbrs(mesh, col, file_name = file_name, plot_dim = 3)
