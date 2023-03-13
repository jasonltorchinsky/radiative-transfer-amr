import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append('../../src')
from dg.mesh import ji_mesh, tools

def test_5(dir_name = 'test_mesh'):
    """
    Visualizes a single column being refined repeatedly.
    """

    dir_name = os.path.join(dir_name, 'test_5')
    os.makedirs(dir_name, exist_ok = True)

    mesh_dir = os.path.join(dir_name, 'mesh')
    os.makedirs(mesh_dir, exist_ok = True)

    nhbrs_dir = os.path.join(dir_name, 'nhbrs')
    os.makedirs(nhbrs_dir, exist_ok = True)

    # Create the original 2-D mesh
    [Lx, Ly] = [3., 2.]
    pbcs     = [True, True]

    mesh = ji_mesh.Mesh([Lx, Ly], pbcs, has_th = True)

    # Refine the single column repeatedly
    ncol_ref = 5
    col = mesh.cols[0]
    for ref in range(0, ncol_ref):
        cell_keys = sorted(col.cells.keys())
        cell = col.cells[cell_keys[-1]]
        mesh.ref_cell(col, cell)
        #mesh.ref_col(col, kind = 'ang')
        
        file_name = os.path.join(mesh_dir, 'mesh_3d_{}.png'.format(ref))
        tools.plot_mesh(mesh, file_name = file_name,
                        label_cells = False, plot_dim = 3)

    for cell_key, cell in sorted(col.cells.items()):
        if cell.is_lf:
            file_name = os.path.join(nhbrs_dir, 'nhbrs_{}.png'.format(cell_key))
            tools.plot_cell_nhbrs(mesh, col, cell, file_name = file_name,
                                  label_cells = False)
    
