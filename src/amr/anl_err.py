import numpy as np

from .Error_Indicator import Error_Indicator
from .hp_steer import hp_steer_col, hp_steer_cell

import dg.quadrature as qd
from dg.projection import push_forward, pull_back


def anl_err(mesh, proj, anl_sol):
    return anl_err_L2(mesh, proj, anl_sol)

def anl_err_L2(mesh, proj, anl_sol,
               ref_form = 'hp', ref_kind = 'all',
               tol_spt = 1.1,   tol_ang = 1.1,
               by_col  = False, by_cell = False):
    """
    The analytic error should be flexible to any ref_form (h-, p-, hp-),
    ref_kind (spt, ang, all), and by_col (spt, ang, ll) or by_cell (ang only).
    
    Here we calculate the L2-error by cell (and column), weighted to be the
    relative error.
    """
    
    err_ind = Error_Indicator(mesh, ref_form = 'hp', ref_kind = 'all',
                              tol_spt = 1.1,   tol_ang = 1.1,
                              by_col  = True,  by_cell = True)
    
    col_items = sorted(mesh.cols.items())
    
    # Relative error is weighted by the integral of the anlytic solution, u'
    u_intg = 0.

    # Track maximum error(s) to calculate hp-steering only where needed
    max_col_err  = 0.
    max_cell_err = 0.
    
    # Calculate the errors
    for col_key, col in col_items:
        if col.is_lf:
            # Column information for quadrature
            [x0, y0, xf, yf] = col.pos[:]
            [dx, dy] = [xf - x0, yf - y0]
            [ndof_x, ndof_y] = col.ndofs[:]
            
            [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                  nnodes_y = ndof_y)
            
            xxf = push_forward(x0, xf, xxb).reshape(ndof_x, 1, 1)
            wx = wx.reshape(ndof_x, 1, 1)
            yyf = push_forward(y0, yf, yyb).reshape(1, ndof_y, 1)
            wy = wy.reshape(1, ndof_y, 1)

            # Loop through cells to calculate error
            col_err    = 0.
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    # Cell information for quadrature
                    [th0, thf] = cell.pos[:]
                    dth = thf - th0
                    [ndof_th]  = cell.ndofs[:]
                    
                    [_, _, _, _, thb, wth] = qd.quad_xyth(nnodes_th = ndof_th)
                    
                    thf = push_forward(th0, thf, thb).reshape(1, 1, ndof_th)
                    wth = wth.reshape(1, 1, ndof_th)

                    # Calculate error
                    uh_cell = proj.cols[col_key].cells[cell_key].vals
                    u_cell  = anl_sol(xxf, yyf, thf)
                    
                    cell_err = np.sum((dx * dy * dth / 8.) * wx * wy * wth * (u_cell - uh_cell)**2)
                    col_err  += cell_err

                    u_intg   += np.sum((dx * dy * dth / 8.) * wx * wy * wth * (u_cell)**2)
                    
                    err_ind.cols[col_key].cells[cell_key].err_ind = cell_err
                    
                    max_cell_err = max(max_cell_err, cell_err)
                    
            max_col_err = max(max_col_err, col_err)
            
            err_ind.cols[col_key].err_ind = col_err

            #err_ind.cols[col_key].cells[cell_key].ref_form = \
            #            hp_steer_cell(mesh, proj, col_key, cell_key)
            #err_ind.cols[col_key].ref_form = hp_steer_col(mesh, proj, col_key)

    # Weight errors to be relative, and calculate hp-steering criteria
    max_col_err  /= u_intg
    max_cell_err /= u_intg
            
    # Weight to be relative error
    for col_key, col in col_items:
        if col.is_lf:
            err_ind.cols[col_key].err_ind /= u_intg

            if err_ind.cols[col_key].err_ind >= 0.:
                return ## CONTINUE FROM HERE!!!!
            
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    err_ind.cols[col_key].cells[cell_key].err_ind /= max_u
                    
                    err_ind.max_err = max(err_ind.max_err,
                                          err_ind.cols[col_key].cells[cell_key].err_ind)

    return err_ind

def anl_err_max(mesh, proj, anl_sol):
    
    col_items = sorted(mesh.cols.items())
    ncols = len(col_items)
    
    err_ind = Error_Indicator(mesh, by_col = True, by_cell = True)

    # To get max-norm relative error, we need the maximal value of anl_sol
    max_u = 0.

    # Get max_norm(u - uh) by column
    for col_key, col in col_items:
        if col.is_lf:
            col_err = 0.
            
            [x0, y0, xf, yf] = col.pos[:]
            [ndof_x, ndof_y] = col.ndofs[:]
            
            [xxb, _, yyb, _, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                  nnodes_y = ndof_y)
            
            xxf = push_forward(x0, xf, xxb).reshape(ndof_x, 1, 1)
            yyf = push_forward(y0, yf, yyb).reshape(1, ndof_y, 1)
            
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, thf] = cell.pos[:]
                    [ndof_th]  = cell.ndofs[:]
                    
                    [_, _, _, _, thb, _] = qd.quad_xyth(nnodes_th = ndof_th)
                    
                    thf = push_forward(th0, thf, thb).reshape(1, 1, ndof_th)
                    
                    uh_cell = proj.cols[col_key].cells[cell_key].vals
                    u_cell  = anl_sol(xxf, yyf, thf)
                    
                    cell_err = np.amax(np.abs(u_cell - uh_cell))
                    col_err = max(col_err, cell_err)
                    
                    err_ind.cols[col_key].cells[cell_key].err_ind = cell_err
                    err_ind.cols[col_key].cells[cell_key].ref_form = \
                        hp_steer_cell(mesh, proj, col_key, cell_key)
                    
                    max_u   = max(max_u, np.amax(np.abs(u_cell)))
            
            err_ind.cols[col_key].err_ind = col_err
            err_ind.cols[col_key].ref_form = hp_steer_col(mesh, proj, col_key)
            
    # Weight to be relative error
    for col_key, col in col_items:
        if col.is_lf:
            err_ind.cols[col_key].err_ind /= max_u
            
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    err_ind.cols[col_key].cells[cell_key].err_ind /= max_u
                    
                    err_ind.max_err = max(err_ind.max_err,
                                          err_ind.cols[col_key].cells[cell_key].err_ind)

    return err_ind
