import argparse
from datetime import datetime
import numpy as np
from time import perf_counter
import os, sys

sys.path.append('../../src')
from utils import print_msg

from tests import test_0, test_1

def main():

    parser_desc = 'Determine which tests to run and where to put output.'
    parser = argparse.ArgumentParser(description = parser_desc)
    parser.add_argument('--dir', nargs = 1, default = 'test_projection',
                        required = False, help = 'Subdirectory to store output')

    help_str = 'Do not run (0) or run (1) all tests (overrides other flags)'
    parser.add_argument('--test_all', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = help_str)

    help_str = ('Do not run (0) or run (1) Test 0 - ' +
                '2-D Prpjection Creation and Plotting')
    parser.add_argument('--test_0', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = help_str)

    help_str = ('Do not run (0) or run (1) Test 1 - ' +
                '3-D Prpjection Creation and Plotting')
    parser.add_argument('--test_1', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = help_str)
    
    args = parser.parse_args()
    ntests = 2
    if args.test_all[0]:
        run_tests = [True] * ntests
    else:
        run_tests = [args.test_0[0], args.test_1[0]
                     ]

    dir_name = args.dir
    os.makedirs(dir_name, exist_ok = True)


    test_funcs = [test_0, test_1]

    for ntest, test_func in enumerate(test_funcs):
        if run_tests[ntest]:
            perf_0 = perf_counter()
            print_msg('Starting Test {}...'.format(ntest))
            
            test_func(dir_name = dir_name)
            
            perf_f = perf_counter()
            perf_diff = perf_f - perf_0
            msg = ('Completed Test {}! ' +
                   'Time Elapsed: {:06.3f} [s]').format(ntest, perf_diff)
            print_msg(msg)
        

if __name__ == '__main__':

    main()
