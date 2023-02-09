import argparse
from datetime import datetime
import numpy as np
from time import perf_counter
import os, sys

sys.path.append('../../src')
from dg.mesh import ji_mesh
from dg.mesh import tools as mesh_tools

from utils import print_msg

from tests import test_0, test_1, test_2, test_3

def main():

    parser_desc = 'Determine which tests to run and where to put output.'
    parser = argparse.ArgumentParser(description = parser_desc)
    parser.add_argument('--dir', nargs = 1, default = 'test_rtdg',
                        required = False, help = 'Subdirectory to store output')
    parser.add_argument('--test_all', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) all tests (overrides other flags)')
    parser.add_argument('--test_0', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 0 - Matrix Mask Construction')
    parser.add_argument('--test_1', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 1 - Mass Matrix Construction')
    parser.add_argument('--test_2', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 2 - Scattering Matrix Construction')
    parser.add_argument('--test_3', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 2 - Convection Matrix Construction')

    args = parser.parse_args()
    ntest = 4
    if args.test_all[0]:
        run_tests = [True] * ntest
    else:
        run_tests = [args.test_0[0], args.test_1[0],
                     args.test_2[0], args.test_3[0]]

    dir_name = args.dir
    os.makedirs(dir_name, exist_ok = True)

    # Create the mesh on which to perform tests.
    Lx = 3
    Ly = 2
    ndofs_x, ndofs_y, ndofs_th = [2, 2, 2]

    # Construct the mesh, with some refinements.
    mesh = ji_mesh.Mesh(Ls = [Lx, Ly],
                        pbcs = [True, False],
                        ndofs = [ndofs_x, ndofs_y, ndofs_th],
                        has_th = True)

    for col_nrefs in range(0, 2):
        mesh.cols[0].ref_col()

    nuni_refs = 1
    for ref in range(0, nuni_refs):
        mesh.ref_mesh()

    mesh_dir  = os.path.join(dir_name, 'mesh')
    os.makedirs(mesh_dir, exist_ok = True)
    mesh_tools.plot_mesh(mesh, file_name = os.path.join(mesh_dir, 'mesh_3d.png'),
                         plot_dim = 3)
    mesh_tools.plot_mesh(mesh, file_name = os.path.join(mesh_dir, 'mesh_2d.png'),
                         plot_dim = 2, label_cells = True)

    # Define background test functions
    if run_tests[0]:
        perf_0 = perf_counter()
        print_msg('Starting Test 0...')
        
        test_0(mesh, dir_name = dir_name)
        
        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        msg = ('Completed Test 0! ' +
               'Time Elapsed: {:06.3f} [s]').format(perf_diff)
        print_msg(msg)

    if run_tests[1]:
        perf_0 = perf_counter()
        print_msg('Starting Test 1...')

        test_1(mesh, dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        msg = ('Completed Test 1! ' +
               'Time Elapsed: {:06.3f} [s]').format(perf_diff)
        print_msg(msg)

    if run_tests[2]:
        perf_0 = perf_counter()
        print_msg('Starting Test 2...')

        test_2(mesh, dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        msg = ('Completed Test 2! ' +
               'Time Elapsed: {:06.3f} [s]').format(perf_diff)
        print_msg(msg)

    if run_tests[3]:
        perf_0 = perf_counter()
        print_msg('Starting Test 3...')

        test_3(mesh, dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        msg = ('Completed Test 3! ' +
               'Time Elapsed: {:06.3f} [s]').format(perf_diff)
        print_msg(msg)
        
    
if __name__ == '__main__':

    main()
