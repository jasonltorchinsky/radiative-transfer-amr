from dg.mesh import ji_mesh
from dg.mesh import tools as mesh_tools


import numpy as np
import sys, getopt
import os
from scipy import sparse

def main(argv):

    print('Actual code lesgoooooo!\n')
    # Set up output directory
    dir_name = 'main_out'
    os.makedirs(dir_name, exist_ok = True)

    # Mesh parameters
    Lx = 16
    Ly = 16
    ndofs_x, ndofs_y, ndofs_a = [2, 2, 2]

    # Construct the mesh, with some refinements.
    mesh = ji_mesh.Mesh(Ls = [Lx, Ly],
                        pbcs = [True, False],
                        ndofs = [ndofs_x, ndofs_y, ndofs_a],
                        is_flat = False)

    col = mesh.cols[0]
    col.ref_col()
    col.ref_col()
    mesh.ref_mesh()
    mesh.ref_mesh()
    
    file_name = 'mesh_0_3d.png'
    file_path = os.path.join(dir_name, file_name)
    mesh_tools.plot_mesh(mesh, file_name = file_path, plot_dim = 3)
    print('Wrote initial mesh to {}\n'.format(file_name))

    print(mesh)


if __name__ == '__main__':

    main(sys.argv[1:])
