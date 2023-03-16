import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append('../../src')
from dg.mesh import ji_mesh, tools

def test_0(dir_name = 'test_mesh'):
    """
    Creates a simple 2-D mesh and visualizes it.
    """

    dir_name = os.path.join(dir_name, 'test_0')
    os.makedirs(dir_name, exist_ok = True)

    # Create the original 2-D mesh
    [Lx, Ly] = [3., 2.]
    pbcs     = [True, True]

    mesh = ji_mesh.Mesh([Lx, Ly], pbcs, has_th = False)
    
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
        mesh.ref_col(col_keys[-1])
        nrefs += 1
        fig, ax = plt.subplots()
        file_name = os.path.join(dir_name, 'mesh_2d_{}.png'.format(nrefs))
        tools.plot_mesh(mesh, ax = ax, file_name = file_name,
                        label_cells = False, plot_dim = 2)

    file_name = os.path.join(dir_name, 'mesh_2d_bdry.png')
    tools.plot_mesh_bdry(mesh, file_name = file_name,
                         label_cells = False, plot_dim = 2)
    
