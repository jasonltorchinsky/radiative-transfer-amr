import argparse
from datetime import datetime
import numpy as np
from time import perf_counter
import os, sys

sys.path.append('../../src')
from utils import print_msg

from tests import test_0, test_1, test_2, test_3

def main():

    parser_desc = 'Determine which tests to run and where to put output.'
    parser = argparse.ArgumentParser(description = parser_desc)
    parser.add_argument('--dir', nargs = 1, default = 'test_mesh',
                        required = False, help = 'Subdirectory to store output')
    parser.add_argument('--test_all', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) all tests (overrides other flags)')
    parser.add_argument('--test_0', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 0 - 2-D Mesh Creation')

    parser.add_argument('--test_1', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 1 - Find Column Neighbors')
    parser.add_argument('--test_2', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 2 - Find Cell Neighbors')

    parser.add_argument('--test_3', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 3 - Find Cell Spatial Neighbors')
    
    args = parser.parse_args()
    ntests = 4
    if args.test_all[0]:
        run_tests = [True] * ntests
    else:
        run_tests = [args.test_0[0], args.test_1[0],
                     args.test_2[0], args.test_3[0]
                     ]

    dir_name = args.dir
    os.makedirs(dir_name, exist_ok = True)

    if run_tests[0]:
        perf_0 = perf_counter()
        print_msg('Starting Test 0...')

        test_0(dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        msg = ('Completed Test 0! ' +
               'Time Elapsed: {:06.3f} [s]').format(perf_diff)
        print_msg(msg)
        
    if run_tests[1]:
        perf_0 = perf_counter()
        print_msg('Starting Test 1...')

        test_1(dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        msg = ('Completed Test 1! ' +
               'Time Elapsed: {:06.3f} [s]').format(perf_diff)
        print_msg(msg)

    if run_tests[2]:
        perf_0 = perf_counter()
        print_msg('Starting Test 2...')

        test_2(dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        msg = ('Completed Test 2! ' +
               'Time Elapsed: {:06.3f} [s]').format(perf_diff)
        print_msg(msg)

    if run_tests[3]:
        perf_0 = perf_counter()
        print_msg('Starting Test 3...')

        test_3(dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        msg = ('Completed Test 3! ' +
               'Time Elapsed: {:06.3f} [s]').format(perf_diff)
        print_msg(msg)
        

if __name__ == '__main__':

    main()
