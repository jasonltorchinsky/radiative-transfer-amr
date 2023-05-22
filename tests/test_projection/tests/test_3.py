import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import os, sys


sys.path.append('../../src')
import dg.quadrature as qd
from dg.projection import push_forward, pull_back, get_f2f_matrix

from utils import print_msg


def test_3(dir_name = 'test_proj'):
    """
    F2F-matrix generation
    """
    
    test_dir = os.path.join(dir_name, 'test_3')
    os.makedirs(dir_name, exist_ok = True)
    
    # Construct the F2F matrix manually.
    # _0 correspond to cell we get basis functions from
    # _1 corresponds to cell we get nodes from
    [x0_0, x1_0] = [0., 2.]
    dx_0 = x1_0 - x0_0

    nhbr_rels = [(0, 's'), (-1, 'l'), (-1, 'u'), (1, 'l'), (1, 'u')]
    
    rng = np.random.default_rng()
    
    do_fail = False
    ntrial = 50
    for trial in range(0, ntrial):
        nhbr_rel = list(rng.choice(nhbr_rels))
        nhbr_rel[0] = int(nhbr_rel[0])
        nhbr_rel = tuple(nhbr_rel)
        if nhbr_rel == (0, 's'):
            [x0_1, x1_1] = [x0_0, x1_0]
        elif nhbr_rel == (-1, 'l'):
            [x0_1, x1_1] = [x0_0, x1_0 + dx_0]
        elif nhbr_rel == (-1, 'u'):
            [x0_1, x1_1] = [x0_0 - dx_0, x1_0]
        elif nhbr_rel == (1, 'l'):
            [x0_1, x1_1] = [x0_0, x1_0 - dx_0/2.]
        elif nhbr_rel == (1, 'u'):
            [x0_1, x1_1] = [x0_0 + dx_0/2., x1_0]
        else:
            print_msg('ERROR: nhbr_rel = {}\n'.format(nhbr_rel))
            quit()
        
        ndof_x_0 = rng.integers(3, 10)
        ndof_x_1 = rng.integers(3, 10)
        
        [xxb_0, w_x_0, _, _, _, _] = qd.quad_xyth(nnodes_x = ndof_x_0)
        [xxb_1, w_x_1, _, _, _, _] = qd.quad_xyth(nnodes_x = ndof_x_1)
        
        xxf_1 = push_forward(x0_1, x1_1, xxb_1)
        xxb_1_0 = pull_back(x0_0, x1_0, xxf_1)
        
        f2f_mat = np.zeros([ndof_x_0, ndof_x_1])
        for ii in range(0, ndof_x_0):
            for pp in range(0, ndof_x_1):
                phi_ip = qd.lag_eval(xxb_0, ii, xxb_1_0[pp])
                if phi_ip >= 1.e-14:
                    f2f_mat[ii, pp] = phi_ip
                    
        f2f_matrix = get_f2f_matrix(dim_str = 'x',
                                    nbasis = ndof_x_0,
                                    nnode  = ndof_x_1,
                                    nhbr_rel = nhbr_rel)

        if np.amax(np.abs(f2f_mat - f2f_matrix)) > 1.e-14:
            do_fail = True
            print_msg('Failed: {}, {}, {}'.format(nhbr_rel, ndof_x_0, ndof_x_1))
            print_msg(f2f_mat)
            print_msg(f2f_matrix)
            print('\n')
            
    if do_fail:
        print_msg('Test failed!\n')
    else:
        print_msg('Test passed!\n')
