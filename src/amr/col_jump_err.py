import numpy as np

from .Error_Indicator import Error_Indicator

import dg.quadrature as qd
from dg.projection import push_forward, pull_back

def col_jump_err(mesh, proj):

    col_items = sorted(mesh.cols.items())
    ncols = len(col_items)

    # Array to store column jump errors
    err_ind = Error_Indicator(mesh, by_col = True, by_cell = False)
    col_intg_ths = {}
    
    # Begin by integrating each column with respect to theta
    for col_key, col in col_items:
        if col.is_lf:                                    
            col_intg_ths[col_key] = intg_col_bdry_th(mesh, proj, col_key)
            
    # Once we have integrated against theta for all cols, we need to integrate
    # the jumps in the spatial dimensions
    for col_key_0, col_0 in col_items:
        if col_0.is_lf:
            [x0_0, y0_0, xf_0, yf_0] = col_0.pos[:]
            [dx_0, dy_0]             = [xf_0 - x0_0, yf_0 - y0_0]
            [ndof_x_0, ndof_y_0]     = col_0.ndofs[:]

            [xxb_0, wx_0, yyb_0, wy_0, _, _] = qd.quad_xyth(nnodes_x = ndof_x_0,
                                                            nnodes_y = ndof_y_0)

            xxf_0 = push_forward(x0_0, xf_0, xxb_0)
            yyf_0 = push_forward(y0_0, yf_0, yyb_0)

            col_err = 0.

            for F in range(0, 4):
                nhbr_keys = list(set(col_0.nhbr_keys[F]))
                
                for col_key_1 in nhbr_keys:
                    if col_key_1 is not None:
                        col_1 = mesh.cols[col_key_1]
                        if col_1.is_lf:
                            [x0_1, y0_1, xf_1, yf_1] = col_1.pos[:]
                            [dx_1, dy_1]             = [xf_1 - x0_1, yf_1 - y0_1]
                            [ndof_x_1, ndof_y_1]     = col_1.ndofs[:]
                            
                            [xxb_1, wx_1, yyb_1, wy_1, _, _] = qd.quad_xyth(nnodes_x = ndof_x_1,
                                                                            nnodes_y = ndof_y_1)
                            xxf_1 = push_forward(x0_1, xf_1, xxb_1)
                            yyf_1 = push_forward(y0_1, yf_1, yyb_1)

                            # Depending on the number of DOFs, we project the
                            # solution on the basis of one column onto the other

                            # Project from the smaller ndof to the larger
                            if (F%2 == 0):
                                dcoeff = dy_0 / 2.
                            else:
                                dcoeff = dx_0 / 2.
                            
                            if (F%2 == 0):
                                if (ndof_y_0 >= ndof_y_1):
                                    col_intg_th_1 = np.zeros([ndof_y_0])
                                    yyb_0_1 = pull_back(y0_1, yf_1, yyf_0)
                                            
                                    for jj in range(0, ndof_y_0):
                                        for qq in range(0, ndof_y_1):
                                            col_intg_th_1[jj] += \
                                                col_intg_ths[col_key_1][(F+2)%4][qq] \
                                                * qd.lag_eval(yyb_1, qq, yyb_0_1[jj])
                                        col_err += dcoeff * wy_0[jj] \
                                            * (col_intg_ths[col_key_0][F][jj] - col_intg_th_1[jj])**2
                                        
                                else:
                                    col_intg_th_0 = np.zeros([ndof_y_1])
                                    yyb_1_0 = pull_back(y0_0, yf_0, yyf_1)
                                            
                                    for qq in range(0, ndof_y_1):
                                        for jj in range(0, ndof_y_0):
                                            col_intg_th_0[qq] += \
                                                col_intg_ths[col_key_0][F][jj] \
                                                * qd.lag_eval(yyb_0, jj, yyb_1_0[qq])
                                        col_err += dcoeff * wy_1[qq] \
                                            * (col_intg_th_0[qq] - col_intg_ths[col_key_1][(F+2)%4][qq])**2
                                        
                            else:
                                if (ndof_x_0 >= ndof_x_1):
                                    col_intg_th_1 = np.zeros([ndof_x_0])
                                    xxb_0_1 = pull_back(x0_1, xf_1, xxf_0)
                                            
                                    for ii in range(0, ndof_x_0):
                                        for pp in range(0, ndof_x_1):
                                            col_intg_th_1[ii] += \
                                                col_intg_ths[col_key_1][(F+2)%4][pp] \
                                                * qd.lag_eval(xxb_1, pp, xxb_0_1[ii])
                                        col_err += dcoeff * wx_0[ii] \
                                            * (col_intg_ths[col_key_0][F][ii] - col_intg_th_1[ii])**2
                                        
                                else:
                                    col_intg_th_0 = np.zeros([ndof_x_1])
                                    xxb_1_0 = pull_back(x0_0, xf_0, xxf_1)
                                            
                                    for pp in range(0, ndof_x_1):
                                        for ii in range(0, ndof_x_0):
                                            col_intg_th_0[pp] += \
                                                col_intg_ths[col_key_0][F][ii] \
                                                * qd.lag_eval(xxb_0, ii, xxb_1_0[pp])
                                        col_err += dcoeff * wx_1[pp] \
                                            * (col_intg_th_0[pp] - col_intg_ths[col_key_1][(F+2)%4][pp])**2
                                        
            perim = 2. * dx_0 + 2. * dy_0
            col_err = np.sqrt((1. / perim) * col_err)
            
            err_ind.cols[col_key_0].err_ind = col_err

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
