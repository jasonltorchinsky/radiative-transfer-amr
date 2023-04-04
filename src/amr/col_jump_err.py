import numpy as np

import dg.quadrature as qd
from dg.projection import push_forward, pull_back

def col_jump_err(mesh, proj):

    col_items = sorted(mesh.cols.items())
    ncols = len(col_items)

    # Array to store column jump errors
    col_errs = {}
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

                # Project the solution on the neighboring columns to the basis
                # of the column of interest
                if (F%2 == 0):
                    col_intg_th_1 = np.zeros([ndof_y_0])
                else:
                    col_intg_th_1 = np.zeros([ndof_x_0])
                    
                
                for col_key_1 in nhbr_keys:
                    if col_key_1 is not None:
                        col_1 = mesh.cols[col_key_1]
                        if col_1.is_lf:
                            [x0_1, y0_1, xf_1, yf_1] = col_1.pos[:]
                            [dx_1, dy_1]             = [xf_1 - x0_1, yf_1 - y0_1]
                            [ndof_x_1, ndof_y_1]     = col_1.ndofs[:]
                            
                            [xxb_1, _, yyb_1, _, _, _] = qd.quad_xyth(nnodes_x = ndof_x_1,
                                                                      nnodes_y = ndof_y_1)

                            xxb_0_1 = pull_back(x0_1, xf_1, xxf_0)
                            yyb_0_1 = pull_back(y0_1, yf_1, yyf_0)

                            if (F%2 == 0):
                                for jj in range(0, ndof_y_0):
                                    for qq in range(0, ndof_y_1):
                                        col_intg_th_1[jj] += \
                                            col_intg_ths[col_key_1][(F+2)%4][qq] * qd.lag_eval(yyb_1, qq, yyb_0_1[jj])

                            else:
                                for ii in range(0, ndof_x_0):
                                    for pp in range(0, ndof_x_1):
                                        col_intg_th_1[ii] += \
                                            col_intg_ths[col_key_1][(F+2)%4][pp] * qd.lag_eval(xxb_1, pp, xxb_0_1[ii])

                # Now integrate the norm in space!
                if (F%2 == 0):
                    for jj in range(0, ndof_y_0):
                        col_err += (dy_0 / 2.) * wy_0[jj] * (col_intg_ths[col_key_0][F][jj] - col_intg_th_1[jj])**2

                else:
                    for ii in range(0, ndof_x_0):
                        col_err += (dx_0 / 2.) * wx_0[ii] * (col_intg_ths[col_key_0][F][ii] - col_intg_th_1[ii])**2

            perim = 2. * dx_0 + 2. * dy_0
            col_err = np.sqrt((1. / perim) * col_err)

            col_errs[col_key_0] = col_err

    return col_errs

def intg_col_bdry_th(mesh, proj, col_key):
    '''
    Integrate the spatial boundary of a column with respect to theta.
    '''

    proj_col = proj.cols[col_key]
    if proj_col.is_lf:
        [x0, y0, xf, yf] = proj_col.pos[:]
        [dx, dy]         = [xf - x0, yf - y0]
        [ndof_x, ndof_y] = proj_col.ndofs[:]
        
        proj_cell_items = sorted(proj_col.cells.items())
        
        # Store the theta integral along each face
        # F = 0 = Right, proceed CCW
        col_intg_th = [np.zeros([ndof_y]), np.zeros([ndof_x]),
                       np.zeros([ndof_y]), np.zeros([ndof_x])]
        
        for F in range(0, 4):
            if (F%2 == 0):
                if (F == 0):
                    x_idx = ndof_x - 1
                elif (F == 2):
                    x_idx = 0
                    
                for cell_key, cell in proj_cell_items:
                    if cell.is_lf:
                        [th0, thf] = cell.pos[:]
                        dth        = thf - th0
                        [ndof_th]  = cell.ndofs[:]
                        
                        [_, _, _, _, _, w_th] = qd.quad_xyth(nnodes_th = ndof_th)
                        w_th *= (dth / 2.) # For integration
                        
                        for jj in range(0, ndof_y):
                            for aa in range(0, ndof_th):
                                col_intg_th[F][jj] += w_th[aa] * cell.vals[x_idx, jj, aa]
                                
                                
            else:
                if (F == 1):
                    y_idx = ndof_y - 1
                elif (F == 3):
                    y_idx = 0
                    
                for cell_key, cell in proj_cell_items:
                    if cell.is_lf:
                        [th0, thf] = cell.pos[:]
                        dth        = thf - th0
                        [ndof_th]  = cell.ndofs[:]
                        
                        [_, _, _, _, _, w_th] = qd.quad_xyth(nnodes_th = ndof_th)
                        w_th *= (dth / 2.) # For integration
                        
                        for ii in range(0, ndof_x):
                            for aa in range(0, ndof_th):
                                col_intg_th[F][ii] += w_th[aa] * cell.vals[ii, y_idx, aa]
                                
                                
            return col_intg_th
