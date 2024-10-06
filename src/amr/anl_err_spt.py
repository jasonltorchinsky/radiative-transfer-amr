import numpy as np

from .Error_Indicator import Error_Indicator
from .hp_steer import hp_steer_col
import dg.quadrature as qd
from dg.projection import push_forward, pull_back

def anl_err_spt(mesh, proj, anl_sol_intg_th, **kwargs):
    
    default_kwargs = {"ref_col"      : True,
                      "col_ref_form" : "hp",
                      "col_ref_kind" : "ang",
                      "col_ref_tol"  : 0.85,
                      "ref_cell"      : False,
                      "cell_ref_form" : None,
                      "cell_ref_kind" : None,
                      "cell_ref_tol"  : None}
    kwargs = {**default_kwargs, **kwargs}
    
    err_ind = Error_Indicator(mesh, **kwargs)
    
    # Relative error is weighted by the maximum value of the angular integral of
    # the analytic solution, u
    u_intg_th_max = 0.
    
    # Track maximum error(s) to calculate hp-steering only where needed
    col_max_err  = 0.
    col_ref_tol  = kwargs["col_ref_tol"]

    # Get max-norm errors
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            # Column information for evaluating angularly-integrated analytic
            # solution
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy]         = [x1 - x0, y1 - y0]
            [ndof_x, ndof_y] = col.ndofs[:]
            
            [xxb, _, yyb, _, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                  nnodes_y = ndof_y)
            
            xxf = push_forward(x0, x1, xxb).reshape(ndof_x, 1)
            yyf = push_forward(y0, y1, yyb).reshape(1, ndof_y)
            
            # Loop through cells to calculate error
            uh_col_intg_th = np.zeros([ndof_x, ndof_y])
            col_err    = 0.
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    # Cell information for quadrature
                    [th0, th1] = cell.pos[:]
                    dth        = th1 - th0
                    [ndof_th]  = cell.ndofs[:]
                    
                    [_, _, _, _, thb, w_th] = qd.quad_xyth(nnodes_th = ndof_th)
                    
                    w_th = w_th.reshape([1, 1, ndof_th])
                    
                    # Calculate error
                    uh_cell = proj.cols[col_key].cells[cell_key].vals
                    uh_col_intg_th += (dth / 2.) * np.sum(w_th * uh_cell, axis = 2) 
                    
            if kwargs["ref_col"]:
                u_col_intg_th = anl_sol_intg_th(xxf, yyf, 0, 2. * np.pi)
                col_err       = np.amax(np.abs(u_col_intg_th - uh_col_intg_th) / (dx * dy * 2. * np.pi))
                
                u_intg_th_max = max(u_intg_th_max, np.amax(np.abs(u_col_intg_th)))
                
                err_ind.cols[col_key].err = col_err
                col_max_err = max(col_max_err, col_err)
                
    # Weight errors to be relative, and calculate hp-steering criteria
    if kwargs["ref_col"]:
        col_max_err  /= u_intg_th_max
        col_ref_thrsh = col_ref_tol * col_max_err
        err_ind.col_max_err = col_max_err
    
    # Weight to be relative error, determine hp-steering
    if kwargs["ref_col"]: # If we"re refining columns
        for col_key, col in col_items:
            if col.is_lf:
                err_ind.cols[col_key].err /= u_intg_th_max
                if err_ind.cols[col_key].err >= col_ref_thrsh: # Does this one need to be refined?
                    if err_ind.cols[col_key].ref_form == "hp": # Does the form of refinement need to be chosen?
                        err_ind.cols[col_key].ref_form = hp_steer_col(mesh, proj, col_key)
                else: # Needn't be refined
                    err_ind.cols[col_key].ref_form = None
                    
    
    return err_ind
