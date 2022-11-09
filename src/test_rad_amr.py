from dg.mesh import ji_mesh
import dg.mesh.tools as mesh_tools
from rad_amr import Projection_2D
from rad_amr.tools import plot_projection_2d

import numpy as np
import sys, getopt
import os

def main(argv):

    print('Performing rad_amr tests...\n')

    dir_name = 'test_rad_amr'
    os.makedirs(dir_name, exist_ok = True)

    # Mesh parameters
    Lx = 2
    Ly = 2
    ndofs_x, ndofs_y, ndofs_a = [2, 2, 2]

    # Construct the mesh, with some refinements.
    mesh = ji_mesh.Mesh(Ls = [Lx, Ly],
                        pbcs = [True, False],
                        ndofs = [ndofs_x, ndofs_y, ndofs_a],
                        has_a = False)

    col = mesh.cols[0]
    for ii in range(0, 2):
        col.ref_col()
    for ii in range(0, 4): 
        mesh.ref_mesh()
    
    file_name = 'mesh_0_3d.png'
    file_path = os.path.join(dir_name, file_name)
    mesh_tools.plot_mesh(mesh, file_name = file_path, plot_dim = 3)
    print('Wrote initial mesh to {}\n'.format(file_name))
    
    def kappa(x, y):
        out = np.exp(-((x - Lx/2)**2 + (y - Ly/2)**2))

        return out
    
    kappah = Projection_2D(mesh, kappa)
    file_name = 'kappah_0.png'
    file_path = os.path.join(dir_name, file_name)
    plot_projection_2d(mesh, kappah, file_name = file_path)
    

if __name__ == '__main__':

    main(sys.argv[1:])
