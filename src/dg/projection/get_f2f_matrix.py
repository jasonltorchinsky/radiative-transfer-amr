import numpy as np

from .push_pull import push_forward, pull_back

import dg.quadrature as qd

f2f_matrices = {}

def get_f2f_matrix(dim_str, nbasis, nnode, nhbr_rel):
    """
    Returns Face-to-face matrix for given degrees of freedom,
    neighbor-configuration.

    The elements of an f2f_matrix look like phi_bar(x_bar), where nbasis is the
    number of basis functions and nnode is the number of nodes.

    nhbr_rel is a list containing information on the neighboring relationship.
    Let "_0" denote the element whose basis functions we are evaluating, and "_1"
    denote the element that is providing the nodes.
    
    nhbr_rel[0] = 0 (same refinement), -1 (_0 more refined), 1 (_0 less refined)
    nhbr_rel[1] - What is the relative position of _0 and _1?
                  If _1 is more refined, is it an "u"pper or "l"ower neighbor?
                  If _0 is more refined, is it an "u"pper or "l"ower neighbor?
                  If equally refined, then they"re the "s"ame
    """
    
    key = (dim_str, nbasis, nnode, nhbr_rel)

    if key in f2f_matrices.keys():
        return f2f_matrices[key]

    # If _0, _1 same-level neighbors with same ndof,
    # then we return identity matrix
    if (nbasis == nnode) and nhbr_rel == (0, "s"):
        f2f_matrices[key] = np.eye(nbasis)
        return f2f_matrices[key]
    
    # Get push-forward endpoints of _1 in relation to normalized endpoints of _0
    if nhbr_rel == (1, "u"):
        [x0_1, x1_1] = [0, 1]
    elif nhbr_rel == (1, "l"):
        [x0_1, x1_1] = [-1, 0]
    elif nhbr_rel == (-1, "u"):
        [x0_1, x1_1] = [-3, 1]
    elif nhbr_rel == (-1, "l"):
        [x0_1, x1_1] = [-1, 3]
    elif nhbr_rel == (0, "s"):
        [x0_1, x1_1] = [-1, 1]
    else:
        msg = "ERROR - Invalid nhbr_rel in get_f2f_matrix: {}".format(nhbr_rel)
        sys.exit(1)
        
    if dim_str == "x":
        [nnb_0, _, _, _, _, _] = qd.quad_xyth(nnodes_x = nbasis)
        [nnb_1, _, _, _, _, _] = qd.quad_xyth(nnodes_x = nnode)
        
    elif dim_str == "y":
        [_, _, nnb_0, _, _, _] = qd.quad_xyth(nnodes_y = nbasis)
        [_, _, nnb_1, _, _, _] = qd.quad_xyth(nnodes_y = nnode)
                
    elif dim_str == "th":
        [_, _, _, _, nnb_0, _] = qd.quad_xyth(nnodes_th = nbasis)
        [_, _, _, _, nnb_1, _] = qd.quad_xyth(nnodes_th = nnode)

    if nhbr_rel[0] == -1: # Basis functions not continuous on whole interval,
        # Must do integral over half interval
        coeff = 0.5
        if nhbr_rel[1] == "l":
            nnb_1 = 0.5 * (nnb_1 - 1.)
        else: # nhbr_rel[1] == "u"
            nnb_1 = 0.5 * (nnb_1 + 1.)
    else:
        coeff = 1
        
    nnf_1 = push_forward(x0_1, x1_1, nnb_1)
    nnb_1_0 = nnf_1[:]
    
    f2f_matrix = np.zeros([nbasis, nnode])
    
    for nn_0 in range(0, nbasis):
        for nn_1 in range(0, nnode):
            phi_ip = qd.lag_eval(nnb_0, nn_0, nnb_1_0[nn_1])
            if np.abs(phi_ip) >= 1.e-14:
                f2f_matrix[nn_0, nn_1] = coeff * phi_ip
            
    f2f_matrices[key] = f2f_matrix
    return f2f_matrices[key]
