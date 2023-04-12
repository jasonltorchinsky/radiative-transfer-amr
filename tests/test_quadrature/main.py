import argparse
from datetime import datetime
import numpy as np
from time import perf_counter
import os

from tests import test_0, test_1, test_2, test_3, test_4, test_5, test_6, \
    test_7, test_8

def main():

    parser_desc = 'Determine which tests to run and where to put output.'
    parser = argparse.ArgumentParser(description = parser_desc)
    parser.add_argument('--dir', nargs = 1, default = 'test_quadrature',
                        required = False, help = 'Subdirectory to store output')
    parser.add_argument('--test_all', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) all tests (overrides other flags)')
    parser.add_argument('--test_0', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 0 - LG/LGL Node Placement')
    parser.add_argument('--test_1', nargs = 1, default = [0],
                        type = int, choices = [0, 1], required = False,
                        help = 'Do not run (0) or run (1) Test 1 - LG/LGL 1D Function Projection Between Bases')
    parser.add_argument('--test_2', nargs = 1, default = [0],
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
    ntests = 9
    if args.test_all[0]:
        run_tests = [True] * ntests
    else:
        run_tests = [args.test_0[0], args.test_1[0],
                     args.test_2[0], args.test_3[0],
                     args.test_4[0], args.test_5[0],
                     args.test_6[0], args.test_7[0],
                     args.test_8[0]]

    dir_name = args.dir
    os.makedirs(dir_name, exist_ok = True)

    if run_tests[0]:
        perf_0 = perf_counter()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = '[{}]: Starting Test 0...\n'.format(current_time)
        print(msg)

        test_0(nnodes = 5, quad_type = 'lg', dir_name = dir_name)
        test_0(nnodes = 5, quad_type = 'lgl', dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = ('[{}]: Completed Test 0! ' +
               'Time Elapsed: {:06.3f} [s]\n').format(current_time, perf_diff)
        print(msg)
        
    if run_tests[1]:
        perf_0 = perf_counter()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = '[{}]: Starting Test 1...\n'.format(current_time)
        print(msg)
        
        test_1(func = f, src_nnodes = 41, trgt_nnodes = 10,
               quad_type = 'lg', dir_name = dir_name)
        test_1(func = f, src_nnodes = 10, trgt_nnodes = 41,
               quad_type = 'lg', dir_name = dir_name)

        test_1(func = f, src_nnodes = 41, trgt_nnodes = 10,
               quad_type = 'lgl', dir_name = dir_name)
        test_1(func = f, src_nnodes = 10, trgt_nnodes = 41,
               quad_type = 'lgl', dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = ('[{}]: Completed Test 1! ' +
               'Time Elapsed: {:06.3f} [s]\n').format(current_time, perf_diff)
        print(msg)

    if run_tests[2]:
        perf_0 = perf_counter()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = '[{}]: Starting Test 2...\n'.format(current_time)
        print(msg)
        
        test_2(func = f, quad_type = 'lg', dir_name = dir_name)
        test_2(func = f, quad_type = 'lgl', dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = ('[{}]: Completed Test 2! ' +
               'Time Elapsed: {:06.3f} [s]\n').format(current_time, perf_diff)
        print(msg)

    if run_tests[3]:
        perf_0 = perf_counter()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = '[{}]: Starting Test 3...\n'.format(current_time)
        print(msg)
        
        test_3(func = f, quad_type = 'lg', dir_name = dir_name)
        test_3(func = f, quad_type = 'lgl', dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = ('[{}]: Completed Test 3! ' +
               'Time Elapsed: {:06.3f} [s]\n').format(current_time, perf_diff)
        print(msg)

    if run_tests[4]:
        perf_0 = perf_counter()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = '[{}]: Starting Test 4...\n'.format(current_time)
        print(msg)
        
        test_4(func = f, func_ddx = dfdx, quad_type = 'lg', dir_name = dir_name)
        test_4(func = f, func_ddx = dfdx, quad_type = 'lgl', dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = ('[{}]: Completed Test 4! ' +
               'Time Elapsed: {:06.3f} [s]\n').format(current_time, perf_diff)
        print(msg)

    if run_tests[5]:
        perf_0 = perf_counter()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = '[{}]: Starting Test 5...\n'.format(current_time)
        print(msg)
        
        test_5(func = f, func_ddx = dfdx, quad_type = 'lg', dir_name = dir_name)
        test_5(func = f, func_ddx = dfdx, quad_type = 'lgl', dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = ('[{}]: Completed Test 5! ' +
               'Time Elapsed: {:06.3f} [s]\n').format(current_time, perf_diff)
        print(msg)

    if run_tests[6]:
        perf_0 = perf_counter()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = '[{}]: Starting Test 6...\n'.format(current_time)
        print(msg)
        
        test_6(func = f, Func = F, quad_type = 'lg', dir_name = dir_name)
        test_6(func = f, Func = F, quad_type = 'lgl', dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = ('[{}]: Completed Test 6! ' +
               'Time Elapsed: {:06.3f} [s]\n').format(current_time, perf_diff)
        print(msg)

    if run_tests[7]:
        perf_0 = perf_counter()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = '[{}]: Starting Test 7...\n'.format(current_time)
        print(msg)
        
        test_7(func = f, Func = F, quad_type = 'lg', dir_name = dir_name)
        test_7(func = f, Func = F, quad_type = 'lgl', dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = ('[{}]: Completed Test 7! ' +
               'Time Elapsed: {:06.3f} [s]\n').format(current_time, perf_diff)
        print(msg)

    if run_tests[8]:
        perf_0 = perf_counter()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = '[{}]: Starting Test 8...\n'.format(current_time)
        print(msg)
        
        test_8(quad_type = 'lg', dir_name = dir_name)
        test_8(quad_type = 'lgl', dir_name = dir_name)

        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        msg = ('[{}]: Completed Test 8! ' +
               'Time Elapsed: {:06.3f} [s]\n').format(current_time, perf_diff)
        print(msg)
    
def f(x):
    """
    Test function for approximation.
    """

    #return (x + 0.25)**2 * np.sin(2 * np.pi * x)

    return (x + 0.25)**6 * np.sin(18 * np.pi * x)

def dfdx(x):
    '''
    Test function for differentiation.
    '''

    #return 2 * (x + 0.25) * (np.sin(2 * np.pi * x)
    #                         + np.pi * (x + 0.25) * np.cos(2 * np.pi * x))

    return 6 * (x + 0.25)**5 * (np.sin(18 * np.pi * x)
                                + 3 * np.pi * (x + 0.25) * np.cos(18 * np.pi * x))

def F(x):
    """
    Test function for integration.
    """

    #return (x + 0.25)**2 * np.sin(2 * np.pi * x)

    term_1 = (- 5120. + 51840. * (np.pi * (1. + 4. * x))**2 \
                      - 87480. * (np.pi * (1. + 4. * x))**4 \
                      + 59049. * (np.pi * (1. + 4. * x))**6) \
            * -2. * np.cos(18. * np.pi * x)
    term_2 = (640. - 2160. * (np.pi * (1. + 4. * x))**2 \
                   + 2187. * (np.pi * (1. + 4. * x))**4) \
            * 72. * np.pi * (1. + 4. * x) * np.sin(18. * np.pi * x)

    return (1. / (8707129344. * np.pi**7)) * (term_1 + term_2)
    
def dg(x):
    '''
    Analytic derivative of test function g.
    '''

    return np.pi * np.cos(np.pi * x)

def G(x):
    '''
    Analytic antiderivatiuve of test function g
    '''

    return -(1.0 / np.pi) * np.cos(np.pi * x)

def f_2D(x, y):
    '''
    Test function for 2D approximation.
    '''

    res = (3 * x + 0.25 * (y - 0.25))**4 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    return res

    
if __name__ == '__main__':

    main()
