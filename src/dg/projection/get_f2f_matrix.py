import numpy as np

import dg.quadrature as qd

f2f_matrices = {}

def get_f2f_matrix(dim_str, nbasis, nnode, nhbr_rel):
    '''
    Returns Face-to-face matrix for given degrees of freedom,
    neighbor-configuration.

    nhbr_rel is a list containing information on the neighboring relationship.
    nhbr_rel[0] = 0 (same refinement), -1 (K more refined), 1 (K less refined)
    nhbr_rel[1] - What is the relative position of K and K'?
                  If K' is more refined, is it an 'u'pper or 'l'ower neighbor?
                  If K is more refined, is it an 'u'pper or 'l'ower neighbor?
                  If equally refined, then they're the 's'ame
    '''
    
    key = (dim_str, nbasis, nnode, nhbr_rel)

    if key in f2f_matrices.keys():
        return f2f_matrices[key]

    # If K, K' same-level neighbors with same ndof, then we return identity matrix
    if (nbasis == nnode) and nhbr_rel == (0, 's'):
        f2f_matrices[key] = np.eye(nbasis)
        return f2f_matrices[key]
    
    # Get normalized endpoints of K'
    if nhbr_rel == (1, 'u'):
        [x0_1, x1_1] = [0, 1]
    elif nhbr_rel == (1, 'l'):
        [x0_1, x1_1] = [-1, 1]
    elif nhbr_rel == (-1, 'u'):
        [x0_1, x1_1] = [-3, 1]
    elif nhbr_rel == (-1, 'l'):
        [x0_1, x1_1] = [-1, 3]
        
    if dim_str == 'x':
        [xx_0, _, _, _, _, _] = qd.quad_xyth(nnodes_x = nbasis)
        [xx_1, _, _, _, _, _] = qd.quad_xyth(nnodes_x = nnode)
        xx_0_1 = pull_back(x0_1, x1_1, xx_0)
        
        f2f_matrix = np.zeros(nbasis, nnode)
        
        for ii_0 in range(0, nbasis):
            for ii_1 in range(0, nnode):
                f2f_matrix[ii_0, ii_1] = qd.lag_eval(xx_0, ii_1, xx_0_1[ii_0])
                
    elif dim_str == 'y':
        [_, _, yy_0, _, _, _] = qd.quad_xyth(nnodes_y = nbasis)
        [_, _, yy_1, _, _, _] = qd.quad_xyth(nnodes_y = nnode)
        yy_0_1 = pull_back(x0_1, x1_1, yy_0)
        
        f2f_matrix = np.zeros(nbasis, nnode)
        
        for jj_0 in range(0, nbasis):
            for jj_1 in range(0, nnode):
                f2f_matrix[jj_0, jj_1] = qd.lag_eval(yy_0, jj_1, yy_0_1[jj_0])
                
    elif dim_str == 'th':
        [_, _, _, _, th_0, _] = qd.quad_xyth(nnodes_th = nbasis)
        [_, _, _, _, th_1, _] = qd.quad_xyth(nnodes_th = nnode)
        th_0_1 = pull_back(x0_1, x1_1, th_0)
        
        f2f_matrix = np.zeros(nbasis, nnode)
        
        for aa_0 in range(0, nbasis):
            for aa_1 in range(0, nnode):
                f2f_matrix[aa_1, aa_0] = qd.lag_eval(th_0, aa_1, th_0_1[aa_0])
        
    f2f_matrices[key] = f2f_matrix
    return f2f_matrices[key]
