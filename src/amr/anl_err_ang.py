import numpy as np

from .Error_Indicator import Error_Indicator
from .hp_steer import hp_steer_cell


import dg.quadrature as qd
from dg.projection import push_forward, pull_back

def anl_err_ang(mesh, proj, anl_sol_intg_xy):
    
    col_items = sorted(mesh.cols.items())
    ncols = len(col_items)
    
    err_ind = Error_Indicator(mesh, by_col = False, by_cell = True)
    
    # To get max-norm relative error, we need the maximal value of anl_sol
    max_u_intg_xy = 0.
    
    # Get max_norm(u - uh) by column
    for col_key, col in col_items:
        if col.is_lf:
            col_err = 0.
            
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy] = [x1 - x0, y1 - y0]
            [ndof_x, ndof_y] = col.ndofs[:]
            
            [xxb, w_x, yyb, w_y, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                      nnodes_y = ndof_y)
            
            xxf = push_forward(x0, x1, xxb).reshape(ndof_x, 1, 1)
            yyf = push_forward(y0, y1, yyb).reshape(1, ndof_y, 1)
            
            w_x = w_x.reshape(ndof_x, 1, 1)
            w_y = w_y.reshape(1, ndof_y, 1)
            
            dcoeff = dx * dy / 4.
           
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, thf] = cell.pos[:]
                    [ndof_th]  = cell.ndofs[:]
                    
                    [_, _, _, _, thb, _] = qd.quad_xyth(nnodes_th = ndof_th)
                    
                    thf = push_forward(th0, thf, thb).reshape(1, 1, ndof_th)
                    
                    uh_cell = proj.cols[col_key].cells[cell_key].vals
                    uh_cell_intg_xy = np.sum(dcoeff * w_x * w_y * uh_cell, axis = (0, 1))
                    
                    u_cell_intg_xy = anl_sol_intg_xy(thf)
                    
                    cell_err = np.amax(np.abs(u_cell_intg_xy - uh_cell_intg_xy))
                    err_ind.cols[col_key].cells[cell_key].err_ind = cell_err
                    
                    err_ind.cols[col_key].cells[cell_key].ref_form = hp_steer_cell(mesh, proj, col_key, cell_key)
                    
                    max_u_intg_xy = max(max_u_intg_xy, np.amax(np.abs(u_cell_intg_xy)))
            
    # Weight to be relative error
    for col_key, col in col_items:
        if col.is_lf:
            cell_items = sorted(col.cells.items())

            for cell_key, cell in cell_items:
                if cell.is_lf:
                    err_ind.cols[col_key].cells[cell_key].err_ind /= max_u_intg_xy

    return err_ind
