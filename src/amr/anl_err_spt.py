import numpy as np

from .Error_Indicator import Error_Indicator
from .hp_steer import hp_steer_col
import dg.quadrature as qd
from dg.projection import push_forward, pull_back

def anl_err_spt(mesh, proj, anl_sol_intg_th):
    
    col_items = sorted(mesh.cols.items())
    ncols = len(col_items)
    
    err_ind = Error_Indicator(mesh, by_col = True, by_cell = False)

    # To get max-norm relative error, we need the maximal value of anl_sol
    max_u_intg_th = 0.

    # Get max_norm(u - uh) by column
    for col_key, col in col_items:
        if col.is_lf:
            col_err = 0.
            
            [x0, y0, xf, yf] = col.pos[:]
            [dx, dy]         = [xf - x0, yf - y0]
            [ndof_x, ndof_y] = col.ndofs[:]
            
            [xxb, _, yyb, _, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                  nnodes_y = ndof_y)
            
            xxf = push_forward(x0, xf, xxb).reshape(ndof_x, 1)
            yyf = push_forward(y0, yf, yyb).reshape(1, ndof_y)
            
            uh_col_intg_th = np.zeros([ndof_x, ndof_y])
           
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, th1] = cell.pos[:]
                    dth        = th1 - th0
                    [ndof_th]  = cell.ndofs[:]
                    
                    [_, _, _, _, thb, w_th] = qd.quad_xyth(nnodes_th = ndof_th)
                    
                    w_th = w_th.reshape([1, 1, ndof_th])
                    
                    uh_cell = proj.cols[col_key].cells[cell_key].vals
                    uh_col_intg_th += (dth / 2.) * np.sum(w_th * uh_cell, axis = 2) 
                    
            u_col_intg_th = anl_sol_intg_th(xxf, yyf, 0, 2. * np.pi)
            col_err       = np.amax(np.abs(u_col_intg_th - uh_col_intg_th) / (dx * dy * 2. * np.pi)) 
            max_u_intg_th = max(max_u_intg_th, np.amax(np.abs(u_col_intg_th)))
            
            err_ind.cols[col_key].err_ind  = col_err
            err_ind.cols[col_key].ref_form = hp_steer_col(mesh, proj, col_key)

    # Weight to be relative error
    for col_key, col in col_items:
        if col.is_lf:
            #err_ind.cols[col_key].err_ind /= max_u_intg_th

            err_ind.max_err = max(err_ind.max_err,
                                  err_ind.cols[col_key].err_ind)

    return err_ind
