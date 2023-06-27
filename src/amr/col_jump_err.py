import numpy as np

from .Error_Indicator import Error_Indicator
from .hp_steer import hp_steer_col

import dg.quadrature as qd
from dg.projection import push_forward, pull_back

def col_jump_err(mesh, proj, **kwargs):

    default_kwargs = {'ref_col'      : True,
                      'col_ref_form' : 'hp',
                      'col_ref_kind' : 'spt',
                      'col_ref_tol'  : 0.85,
                      'ref_cell'      : False,
                      'cell_ref_form' : None,
                      'cell_ref_kind' : None,
                      'cell_ref_tol'  : None}
    kwargs = {**default_kwargs, **kwargs}
    
    err_ind = Error_Indicator(mesh, **kwargs)
    
    col_items = sorted(mesh.cols.items())
    
    # Track maximum error(s) to calculate hp-steering only where needed
    col_max_err  = 0.
    col_ref_tol  = kwargs['col_ref_tol']
    
    # Begin by angularly-integrating each column
    col_intg_ths = {}
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:                                    
            col_intg_ths[col_key] = intg_col_bdry_th(mesh, proj, col_key)
            
    # Once we have angularly-integrated along spatial faces of each col, we
    # spatially-integrate the jumps
    for col_key_0, col_0 in col_items:
        if col_0.is_lf:
            # Column information for quadrature
            [x0_0, y0_0, x1_0, y1_0] = col_0.pos[:]
            [dx_0, dy_0]             = [x1_0 - x0_0, y1_0 - y0_0]
            [ndof_x_0, ndof_y_0]     = col_0.ndofs[:]
            
            [xxb_0, wx_0, yyb_0, wy_0, _, _] = qd.quad_xyth(nnodes_x = ndof_x_0,
                                                            nnodes_y = ndof_y_0)
            
            xx1_0 = push_forward(x0_0, x1_0, xxb_0)
            yyf_0 = push_forward(y0_0, y1_0, yyb_0)

            # Loop through faces to calculate error
            col_err = 0.
            for F in range(0, 4):
                nhbr_keys = list(set(col_0.nhbr_keys[F]))
                
                col_intg_th_0 = col_intg_ths[col_key_0][F]
                
                for col_key_1 in nhbr_keys:
                    if col_key_1 is not None:
                        col_1 = mesh.cols[col_key_1]
                        if col_1.is_lf:
                            # Column information for quadrature
                            [x0_1, y0_1, x1_1, y1_1] = col_1.pos[:]
                            [dx_1, dy_1]             = [x1_1 - x0_1, y1_1 - y0_1]
                            [ndof_x_1, ndof_y_1]     = col_1.ndofs[:]
                            
                            [xxb_1, wx_1, yyb_1, wy_1, _, _] = qd.quad_xyth(nnodes_x = ndof_x_1,
                                                                            nnodes_y = ndof_y_1)
                            
                            col_intg_th_1 = col_intg_ths[col_key_1][(F+2)%4]
                            
                            if (F%2 == 0):
                                # Integrate over whichever face is smaller
                                if dy_0 >= dy_1:
                                    dcoeff = dy_1 / 2
                                    
                                    # Use quadrature rule for whichever column
                                    # has more nodes
                                    if (ndof_y_0 >= ndof_y_1):
                                        # Project col_0
                                        yyf_0_1 = push_forward(y0_1, y1_1, yyb_0)
                                        yyb_0_1_0 = pull_back(y0_0, y1_0, yyf_0_1)
                                        psi_jjp_matrix = np.zeros([ndof_y_0, ndof_y_0])
                                        for jj in range(0, ndof_y_0):
                                            for jj_p in range(0, ndof_y_0):
                                                psi_jjp_matrix[jj, jj_p] = \
                                                    qd.lag_eval(yyb_0, jj, yyb_0_1_0[jj_p])
                                                
                                        col_intg_th_0_0 = np.zeros(ndof_y_0)
                                        for jj_p in range(0, ndof_y_0):
                                            for jj in range(0, ndof_y_0):
                                                col_intg_th_0_0[jj_p] += col_intg_th_0[jj] * psi_jjp_matrix[jj, jj_p]
                                                
                                        # Project col_1
                                        psi_qjp_matrix = np.zeros([ndof_y_1, ndof_y_0])
                                        for qq in range(0, ndof_y_1):
                                            for jj_p in range(0, ndof_y_0):
                                                psi_qjp_matrix[qq, jj_p] = \
                                                    qd.lag_eval(yyb_1, qq, yyb_0[jj_p])
                                                
                                        col_intg_th_1_0 = np.zeros(ndof_y_0)
                                        for jj_p in range(0, ndof_y_0):
                                            for qq in range(0, ndof_y_1):
                                                col_intg_th_1_0[jj_p] += col_intg_th_1[qq] * psi_qjp_matrix[qq, jj_p]
                                                
                                        col_err += dcoeff * np.sum(wy_0 * (col_intg_th_0_0 - col_intg_th_1_0)**2)
                                                
                                    else: # ndof_y_0 < ndof_y_1
                                        # Project col_0
                                        yyf_1 = push_forward(y0_1, y1_1, yyb_1)
                                        yyb_1_0 = pull_back(y0_0, y1_0, yyf_1)
                                        psi_jqp_matrix = np.zeros([ndof_y_0, ndof_y_1])
                                        for jj in range(0, ndof_y_0):
                                            for qq_p in range(0, ndof_y_1):
                                                psi_jqp_matrix[jj, qq_p] = \
                                                    qd.lag_eval(yyb_0, jj, yyb_1_0[qq_p])
                                                                                                
                                        col_intg_th_0_1 = np.zeros(ndof_y_1)
                                        for qq_p in range(0, ndof_y_1):
                                            for jj in range(0, ndof_y_0):
                                                col_intg_th_0_1[qq_p] += col_intg_th_0[jj] * psi_jqp_matrix[jj, qq_p]
                                                
                                        col_err += dcoeff * np.sum(wy_1 * (col_intg_th_0_1 - col_intg_th_1)**2)
                                        
                                else: # dy_0 < dy_1:
                                    dcoeff = dy_0 / 2
                                    
                                    # Use quadrature rule for whichever column
                                    # has more nodes
                                    if (ndof_y_0 >= ndof_y_1):
                                        # Perform projection for col_1
                                        yyf_0 = push_forward(y0_0, y1_0, yyb_0)
                                        yyb_0_1 = pull_back(y0_1, y1_1, yyf_0)
                                        psi_qjp_matrix = np.zeros([ndof_y_1, ndof_y_0])
                                        for qq in range(0, ndof_y_1):
                                            for jj_p in range(0, ndof_y_0):
                                                psi_qjp_matrix[qq, jj_p] = \
                                                    qd.lag_eval(yyb_1, qq, yyb_0_1[jj_p])
                                                                                                
                                        col_intg_th_1_0 = np.zeros(ndof_y_0)
                                        for jj_p in range(0, ndof_y_0):
                                            for qq in range(0, ndof_y_1):
                                                col_intg_th_0_0[jj_p] += col_intg_th_1[qq] * psi_qjp_matrix[qq, jj_p]
                                                
                                        col_err += dcoeff * np.sum(wy_0 * (col_intg_th_0 - col_intg_th_1_0)**2)
                                        
                                    else: # ndof_y_0 < ndof_y_1
                                        # Perform projection for col_0
                                        psi_jqp_matrix = np.zeros([ndof_y_0, ndof_y_1])
                                        for jj in range(0, ndof_y_0):
                                            for qq_p in range(0, ndof_y_1):
                                                psi_qjp_matrix[jj, qq_p] = \
                                                    qd.lag_eval(yyb_0, jj, yyb_1[qq_p])
                                                
                                        col_intg_th_0_1 = np.zeros(ndof_y_1)
                                        for qq_p in range(0, ndof_y_1):
                                            for jj in range(0, ndof_y_0):
                                                col_intg_th_0_1[qq_p] += col_intg_th_0[jj] * psi_jqp_matrix[jj, qq_p]
                                                
                                        # Perform projection for col_1
                                        yyf_1_0 = push_forward(y0_0, y1_0, yyb_1)
                                        yyb_1_0_1 = pull_back(y0_1, y1_1, yyf_1_0)
                                        psi_qqp_matrix = np.zeros([ndof_y_1, ndof_y_1])
                                        for qq in range(0, ndof_y_1):
                                            for qq_p in range(0, ndof_y_1):
                                                psi_qqp_matrix[qq, qq_p] = \
                                                    qd.lag_eval(yyb_1, qq, yyb_1_0_1[qq_p])
                                                
                                        col_intg_th_1_1 = np.zeros(ndof_y_1)
                                        for qq_p in range(0, ndof_y_1):
                                            for qq in range(0, ndof_y_1):
                                                col_intg_th_1_1[qq_p] += col_intg_th_1[qq] * psi_qqp_matrix[qq, qq_p]
                                                
                                        col_err += dcoeff * np.sum(wy_1 * (col_intg_th_0_1 - col_intg_th_1_1)**2)
                                        
                            else: # (F%2) == 1
                                # Integrate over whichever face is smaller
                                if dx_0 >= dx_1:
                                    dcoeff = dx_1 / 2
                                    
                                    # Use quadrature rule for whichever column
                                    # has more nodes
                                    if (ndof_x_0 >= ndof_x_1):
                                        # Project col_0
                                        xxf_0_1 = push_forward(y0_1, y1_1, xxb_0)
                                        xxb_0_1_0 = pull_back(y0_0, y1_0, xxf_0_1)
                                        phi_iip_matrix = np.zeros([ndof_x_0, ndof_x_0])
                                        for ii in range(0, ndof_x_0):
                                            for ii_p in range(0, ndof_x_0):
                                                phi_iip_matrix[ii, ii_p] = \
                                                    qd.lag_eval(xxb_0, ii, xxb_0_1_0[ii_p])
                                                
                                        col_intg_th_0_0 = np.zeros(ndof_x_0)
                                        for ii_p in range(0, ndof_x_0):
                                            for ii in range(0, ndof_x_0):
                                                col_intg_th_0_0[ii_p] += col_intg_th_0[ii] * phi_iip_matrix[ii, ii_p]
                                                
                                        # Project col_1
                                        phi_pip_matrix = np.zeros([ndof_x_1, ndof_x_0])
                                        for pp in range(0, ndof_x_1):
                                            for ii_p in range(0, ndof_x_0):
                                                phi_pip_matrix[pp, ii_p] = \
                                                    qd.lag_eval(xxb_1, pp, xxb_0[ii_p])
                                                
                                        col_intg_th_1_0 = np.zeros(ndof_x_0)
                                        for ii_p in range(0, ndof_x_0):
                                            for pp in range(0, ndof_x_1):
                                                col_intg_th_1_0[ii_p] += col_intg_th_1[pp] * phi_pip_matrix[pp, ii_p]
                                                
                                        col_err += dcoeff * np.sum(wx_0 * (col_intg_th_0_0 - col_intg_th_1_0)**2)
                                                
                                    else: # ndof_x_0 < ndof_x_1
                                        # Project col_0
                                        xxf_1 = push_forward(y0_1, y1_1, xxb_1)
                                        xxb_1_0 = pull_back(y0_0, y1_0, xxf_1)
                                        phi_ipp_matrix = np.zeros([ndof_x_0, ndof_x_1])
                                        for ii in range(0, ndof_x_0):
                                            for pp_p in range(0, ndof_x_1):
                                                phi_ipp_matrix[ii, pp_p] = \
                                                    qd.lag_eval(xxb_0, ii, xxb_1_0[pp_p])
                                                                                                
                                        col_intg_th_0_1 = np.zeros(ndof_x_1)
                                        for pp_p in range(0, ndof_x_1):
                                            for ii in range(0, ndof_x_0):
                                                col_intg_th_0_1[pp_p] += col_intg_th_0[ii] * phi_ipp_matrix[ii, pp_p]
                                                
                                        col_err += dcoeff * np.sum(wx_1 * (col_intg_th_0_1 - col_intg_th_1)**2)
                                        
                                else: # dx_0 < dx_1:
                                    dcoeff = dx_0 / 2
                                    
                                    # Use quadrature rule for whichever column
                                    # has more nodes
                                    if (ndof_x_0 >= ndof_x_1):
                                        # Perform projection for col_1
                                        xxf_0 = push_forward(y0_0, y1_0, xxb_0)
                                        xxb_0_1 = pull_back(y0_1, y1_1, xxf_0)
                                        phi_pip_matrix = np.zeros([ndof_x_1, ndof_x_0])
                                        for pp in range(0, ndof_x_1):
                                            for ii_p in range(0, ndof_x_0):
                                                phi_pip_matrix[pp, ii_p] = \
                                                    qd.lag_eval(xxb_1, pp, xxb_0_1[ii_p])
                                                                                                
                                        col_intg_th_1_0 = np.zeros(ndof_x_0)
                                        for ii_p in range(0, ndof_x_0):
                                            for pp in range(0, ndof_x_1):
                                                col_intg_th_0_0[ii_p] += col_intg_th_1[pp] * phi_pip_matrix[pp, ii_p]
                                                
                                        col_err += dcoeff * np.sum(wx_0 * (col_intg_th_0 - col_intg_th_1_0)**2)
                                        
                                    else: # ndof_x_0 < ndof_x_1
                                        # Perform projection for col_0
                                        phi_ipp_matrix = np.zeros([ndof_x_0, ndof_x_1])
                                        for ii in range(0, ndof_x_0):
                                            for pp_p in range(0, ndof_x_1):
                                                phi_pip_matrix[ii, pp_p] = \
                                                    qd.lag_eval(xxb_0, ii, xxb_1[pp_p])
                                                
                                        col_intg_th_0_1 = np.zeros(ndof_x_1)
                                        for pp_p in range(0, ndof_x_1):
                                            for ii in range(0, ndof_x_0):
                                                col_intg_th_0_1[pp_p] += col_intg_th_0[ii] * phi_ipp_matrix[ii, pp_p]
                                                
                                        # Perform projection for col_1
                                        xxf_1_0 = push_forward(y0_0, y1_0, xxb_1)
                                        xxb_1_0_1 = pull_back(y0_1, y1_1, xxf_1_0)
                                        phi_ppp_matrix = np.zeros([ndof_x_1, ndof_x_1])
                                        for pp in range(0, ndof_x_1):
                                            for pp_p in range(0, ndof_x_1):
                                                phi_ppp_matrix[pp, pp_p] = \
                                                    qd.lag_eval(xxb_1, pp, xxb_1_0_1[pp_p])
                                                
                                        col_intg_th_1_1 = np.zeros(ndof_x_1)
                                        for pp_p in range(0, ndof_x_1):
                                            for pp in range(0, ndof_x_1):
                                                col_intg_th_1_1[pp_p] += col_intg_th_1[pp] * phi_ppp_matrix[pp, pp_p]
                                                
                                        col_err += dcoeff * np.sum(wx_1 * (col_intg_th_0_1 - col_intg_th_1_1)**2)
                                        
            perim = 2. * dx_0 + 2. * dy_0
            col_err = np.sqrt((1. / perim) * col_err)
            col_max_err = max(col_max_err, col_err)
            
            if kwargs['ref_col']:
                err_ind.cols[col_key_0].err = col_err
                
    # Weight to be relative error, determine hp-steering
    if kwargs['ref_col']:
        err_ind.col_max_err = col_max_err
        col_ref_thrsh = col_ref_tol * col_max_err
        
        for col_key, col in col_items:
            if col.is_lf:
                if err_ind.cols[col_key].err >= col_ref_thrsh: # Does this one need to be refined?
                    if err_ind.cols[col_key].ref_form == 'hp': # Does the form of refinement need to be chosen?
                        err_ind.cols[col_key].ref_form = hp_steer_col(mesh, proj, col_key)
                else: # Needn't be refined
                    err_ind.cols[col_key].ref_form = None
                    
    return err_ind

def intg_col_bdry_th(mesh, proj, col_key):
    '''
    Integrate the spatial boundary of a column with respect to theta.
    '''

    proj_col = proj.cols[col_key]
    if proj_col.is_lf:
        [ndof_x, ndof_y] = proj_col.ndofs[:]
        
        proj_cell_items = sorted(proj_col.cells.items())
        
        # Store the theta integral along each face
        # F = 0 = Right, proceed CCW
        col_intg_th = [np.zeros([ndof_y]), np.zeros([ndof_x]),
                       np.zeros([ndof_y]), np.zeros([ndof_x])]

        for cell_key, cell in proj_cell_items:
            if cell.is_lf:
                [th0, thf] = cell.pos[:]
                dth        = thf - th0
                [ndof_th]  = cell.ndofs[:]

                [_, _, _, _, _, w_th] = qd.quad_xyth(nnodes_th = ndof_th)
                
        
                for F in range(0, 4):
                    if (F%2 == 0):
                        if (F == 0):
                            x_idx = ndof_x - 1
                        elif (F == 2):
                            x_idx = 0

                        for jj in range(0, ndof_y):
                            col_intg_th[F][jj] += \
                                (dth / 2.) * np.sum(w_th * cell.vals[x_idx, jj, :])
                                 
                    else:
                        if (F == 1):
                            y_idx = ndof_y - 1
                        elif (F == 3):
                            y_idx = 0

                        for ii in range(0, ndof_x):
                            col_intg_th[F][ii] += \
                                (dth / 2.) * np.sum(w_th * cell.vals[ii, y_idx, :])
                                
        return col_intg_th
