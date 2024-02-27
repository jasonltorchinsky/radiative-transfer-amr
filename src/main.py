from dg.mesh import ji_mesh
from dg.mesh import tools as mesh_tools

from rad_amr import rtdg_amr

from utils import print_msg

import numpy as np
import sys, getopt
import os

def main(argv):

    print_msg('Starting execution...')
    
    # Set up output directory
    dir_name = 'main_out'
    os.makedirs(dir_name, exist_ok = True)

    print_msg('Creating initial mesh...')
    # Mesh parameters
    Lx = 16
    Ly = 16
    ndofs_x, ndofs_y, ndofs_th = [2, 2, 2]

    # Construct the mesh, with some refinements.
    mesh = ji_mesh.Mesh(Ls = [Lx, Ly],
                        pbcs = [True, False],
                        ndofs = [ndofs_x, ndofs_y, ndofs_th],
                        has_th = True)

    col = mesh.cols[0]
    col.ref_col()
    col.ref_col()
    mesh.ref_mesh()
    mesh.ref_mesh()

    mesh_tools.write_mesh(mesh, "out.json")
    

if __name__ == '__main__':

    main(sys.argv[1:])
