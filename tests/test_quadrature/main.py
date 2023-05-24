import argparse
from datetime import datetime
import numpy as np
from time import perf_counter
import os, sys

sys.path.append('../../src')
from utils import print_msg

from tests import test_0, test_1, test_2

def main():

    parser_desc = 'Determine which tests to run and where to put output.'
    parser = argparse.ArgumentParser(description = parser_desc)

    help_str = 'Subdirectory to store output'
    parser.add_argument('--dir', nargs = 1, default = 'test_quadrature',
                        required = False, help = help_str)

    help_str = 'Do not run (0) or run (1) all tests (overrides other flags)'
    parser.add_argument('--test_all', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = help_str)

    help_str = 'Do not run (0) or run (1) Test 0 - Node Placement'
    parser.add_argument('--test_0', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = help_str)

    help_str = 'Do not run (0) or run (1) Test 1 - 1D Function Approximation - Visual'
    parser.add_argument('--test_1', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = help_str)

    help_str = 'Do not run (0) or run (1) Test 2 - Quadrature Rule Convergence'
    parser.add_argument('--test_2', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = help_str)
    
    parser.add_argument('--test_33', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 2 - LG/LGL 1D Function Projection Comparison')
    parser.add_argument('--test_3', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 3 - LG/LGL 1D Function Projection Accuracy')
    parser.add_argument('--test_4', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 4 - LG/LGL 1D Derivative Function Projection Comparison')
    parser.add_argument('--test_5', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 5 - LG/LGL 1D Function Derivative Projection Accuracy')
    parser.add_argument('--test_6', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 6 - LG/LGL 1D Function Antidifferentiation')
    parser.add_argument('--test_7', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 7 - LG/LGL 1D Function Integration')

    help_str = 'Do not run (0) or run (1) Test 8 - LG/LGL Dirac Delta Approximation'
    parser.add_argument('--test_8', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = help_str)

    args = parser.parse_args()
    ntests = 3
    if args.test_all[0]:
        run_tests = [True] * ntests
    else:
        run_tests = [args.test_0[0], args.test_1[0],
                     args.test_2[0]]

    dir_name = args.dir
    os.makedirs(dir_name, exist_ok = True)

    quad_types = ['lg', 'lgr', 'lgl', 'uni']
    test_funcs = [test_0, test_1, test_2]
    
    for ntest, test_func in enumerate(test_funcs):
        if run_tests[ntest]:
            perf_0 = perf_counter()
            print_msg('Starting Test {}...\n'.format(ntest))

            for quad_type in quad_types:
                test_func(quad_type = quad_type, dir_name = dir_name)
                
            perf_f = perf_counter()
            perf_diff = perf_f - perf_0
            msg = ('Completed Test {}! ' +
                   'Time Elapsed: {:08.3f} [s]\n').format(ntest, perf_diff)
            print_msg(msg)
       
if __name__ == '__main__':

    main()
