import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append("../../src")
import dg.mesh as ji_mesh
from dg.mesh.utils import plot_mesh, plot_nhbrs

def test_3(dir_name = "test_mesh"):
    """
    Mixed non-uniform angular/spatial refinement, and neighbors for cells.
    """

    dir_name = os.path.join(dir_name, "test_3")
    os.makedirs(dir_name, exist_ok = True)

    mesh_dir  = os.path.join(dir_name, "mesh")
    os.makedirs(mesh_dir, exist_ok = True)
    
    col_nhbrs_dir = os.path.join(dir_name, "col_nhbrs")
    os.makedirs(col_nhbrs_dir, exist_ok = True)

    cell_nhbrs_dir = os.path.join(dir_name, "cell_nhbrs")
    os.makedirs(cell_nhbrs_dir, exist_ok = True)

    # Create the original 2-D mesh
    [Lx, Ly] = [3, 2]
    pbcs     = [True, False]

    mesh = ji_mesh.Mesh([Lx, Ly], pbcs, has_th = True)

    # Randomly refine the mesh a whole bunch of times.
    rng = np.random.default_rng()
    nref_max = 8
    for nref in range(1, nref_max + 1):
        col_keys = list(mesh.cols.keys())
        col_key = rng.choice(col_keys)
        
        col = mesh.cols[col_key]
        cell_keys = list(col.cells.keys())
        cell_key = rng.choice(cell_keys)
        
        mesh.ref_cell(col_key, cell_key, form = "h")
        mesh.ref_col(col_key, kind = "spt", form = "h")
        
        file_name = os.path.join(mesh_dir, "mesh_{}.png".format(nref))
        plot_mesh(mesh, ax = None, file_name = file_name,
                  plot_dim = 3, plot_style = "flat")

    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            file_str  = "col_{}_nhbrs.png".format(col_key)
            file_name = os.path.join(col_nhbrs_dir, file_str)
            plot_nhbrs(mesh, col_key, file_name = file_name)
                
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                file_str  = "cell_{}_{}_nhbrs.png".format(col_key, cell_key)
                file_name = os.path.join(cell_nhbrs_dir, file_str)
                plot_nhbrs(mesh, col_key, cell_key, file_name = file_name)
