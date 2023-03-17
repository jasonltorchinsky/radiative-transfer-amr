import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append('../../src')
from dg.mesh import ji_mesh, tools

def test_0(dir_name = 'test_mesh'):
    """
    Uniform refinement of 2-D mesh and boundary check.
    """

    dir_name = os.path.join(dir_name, 'test_0')
    os.makedirs(dir_name, exist_ok = True)

    mesh_dir  = os.path.join(dir_name, 'mesh')
    os.makedirs(mesh_dir, exist_ok = True)

    # Create the original 2-D mesh
    [Lx, Ly] = [3., 2.]
    pbcs     = [False, True]

    mesh = ji_mesh.Mesh([Lx, Ly], pbcs, has_th = False)
    
    # Refine the mesh some so we have a more interesting plot
    nrefs = 0
    
    file_name = os.path.join(mesh_dir, 'mesh_2d_{}.png'.format(nrefs))
    tools.plot_mesh(mesh, ax = None, file_name = file_name,
                    label_cells = True, plot_dim = 2)
    
    nuni_ref = 4
    for ref in range(0, nuni_ref):
        mesh.ref_mesh(kind = 'spt')
        nrefs += 1
        
        file_name = os.path.join(mesh_dir, 'mesh_2d_{}.png'.format(nrefs))
        tools.plot_mesh(mesh, ax = None, file_name = file_name,
                        label_cells = (nrefs <= 3), plot_dim = 2)

    file_name = os.path.join(mesh_dir, 'mesh_2d_bdry.png')
    tools.plot_mesh_bdry(mesh, file_name = file_name,
                         label_cells = False, plot_dim = 2)
    
