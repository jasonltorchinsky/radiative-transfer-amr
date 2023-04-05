import numpy as np

from .Error_Indicator import Error_Indicator

import dg.quadrature as qd
from dg.projection import push_forward, pull_back

def anl_err(mesh, proj, anl_sol):

    col_items = sorted(mesh.cols.items())
    ncols = len(col_items)

    # Array to store column jump errors
    err_ind = Error_Indicator(mesh, by_col = True, by_cell = False)
    col_intg_ths = {}
    
            
    # Once we have integrated against theta for all cols, we need to integrate
    # the jumps in the spatial dimensions
    for col_key, col in col_items:
        if col.is_lf:
            col_err = 0.
            
            [x0, y0, xf, yf] = col.pos[:]
            [dx, dy]         = [xf - x0, yf - y0]
            [ndof_x, ndof_y] = col.ndofs[:]

            [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                    nnodes_y = ndof_y)

            xxf = push_forward(x0, xf, xxb)
            yyf = push_forward(y0, yf, yyb)

            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, thf] = cell.pos[:]
                    dth        = thf - th0
                    [ndof_th]  = cell.ndofs[:]
                    cell_vals  = proj.cols[col_key].cells[cell_key].vals
                    
                    [_, _, _, _, thb, w_th] = qd.quad_xyth(nnodes_th = ndof_th)

                    thf = push_forward(th0, thf, thb)

                    for ii in range(0, ndof_x):
                        x_i = xxf[ii]
                        for jj in range(0, ndof_y):
                            y_j = yyf[jj]
                            for aa in range(0, ndof_th):
                                th_a = thf[aa]
                                u_ija = anl_sol(x_i, y_j, th_a)
                                
                                uh_ija = cell_vals[ii, jj, aa]
                                
                                col_err = max(col_err, np.abs(u_ija - uh_ija))
            
            err_ind.cols[col_key].err_ind = col_err

    return err_ind
