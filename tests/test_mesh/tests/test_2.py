import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append("../../src")
import dg.mesh as ji_mesh
import dg.mesh.utils

def test_2(dir_name = "test_mesh"):
    """
    Column spatial refinement, and spatial neighbors for columns.
    """

    dir_name = os.path.join(dir_name, "test_2")
    os.makedirs(dir_name, exist_ok = True)

    mesh_dir  = os.path.join(dir_name, "mesh")
    os.makedirs(mesh_dir, exist_ok = True)
    
    nhbrs_dir = os.path.join(dir_name, "nhbrs")
    os.makedirs(nhbrs_dir, exist_ok = True)

    # Create the original 2-D mesh
    [Lx, Ly] = [3, 2]
    pbcs     = [True, False]

    mesh = ji_mesh.Mesh([Lx, Ly], pbcs, has_th = True)

    # Refine the mesh some so we have a more interesting plot
    nrefs = 0
    
    file_name = os.path.join(mesh_dir, "mesh_2d_{}.png".format(nrefs))
    ji_mesh.utils.plot_mesh(mesh, ax = None, file_name = file_name,
                            label_cells = (nrefs <= 3), plot_dim = 2)

    # Uniform spatial refinements
    nunispt_ref = 2
    for ref in range(0, nunispt_ref):
        mesh.ref_mesh(kind = "spt")

        nrefs += 1

        file_name = os.path.join(mesh_dir, "mesh_2d_{}.png".format(nrefs))
        ji_mesh.utils.plot_mesh(mesh, ax = None, file_name = file_name,
                                label_cells = (nrefs <= 3), plot_dim = 2)
        
    
    ncolspt_ref = 3
    for ref in range(0, ncolspt_ref):
        col_keys = sorted(list(mesh.cols.keys()))
        mesh.ref_col(col_keys[-1], kind = "spt")
        nrefs += 1
        
        file_name = os.path.join(mesh_dir, "mesh_2d_{}.png".format(nrefs))
        ji_mesh.utils.plot_mesh(mesh, ax = None, file_name = file_name,
                                label_cells = (nrefs <= 3), plot_dim = 2)
        
    file_name = os.path.join(mesh_dir, "mesh_2d_bdry.png")
    ji_mesh.utils.plot_mesh_bdry(mesh, file_name = file_name,
                         label_cells = False, plot_dim = 2)
    
    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            file_str  = "col_{}_nhbrs.png".format(col_key)
            file_name = os.path.join(nhbrs_dir, file_str)
            ji_mesh.utils.plot_col_nhbrs(mesh, col, file_name = file_name)
