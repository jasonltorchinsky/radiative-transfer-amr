import numpy as np

from .Error_Indicator import Error_Indicator
from .hp_steer import hp_steer_cell, hp_steer_col

import dg.quadrature as qd
from dg.projection import push_forward, pull_back

def anl_err_ang(mesh, proj, anl_sol_intg_xy, **kwargs):
    """
    Calculate the max-error of the spatial-integral of the analytic solution and
    numerical solution by column and cell, weighted to be average relative error.
    """

    default_kwargs = {'ref_col'      : True,
                      'col_ref_form' : 'hp',
                      'col_ref_kind' : 'ang',
                      'col_ref_tol'  : 0.85,
                      'ref_cell'      : True,
                      'cell_ref_form' : 'hp',
                      'cell_ref_kind' : 'ang',
                      'cell_ref_tol'  : 0.85}
    kwargs = {**default_kwargs, **kwargs}
    
    err_ind = Error_Indicator(mesh, **kwargs)
    
    col_items = sorted(mesh.cols.items())
    
    # Relative error is weighted by the maximum value of the spatial integral of
    # the analytic solution, u
    u_intg_xy_max = 0.
    
    # Track maximum error(s) to calculate hp-steering only where needed
    col_max_err  = 0.
    col_ref_tol  = kwargs['col_ref_tol']
    cell_max_err = 0.
    cell_ref_tol = kwargs['cell_ref_tol']

    # Get max-norm errors
    for col_key, col in col_items:
        if col.is_lf:
            # Column information for quadrature
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy]         = [x1 - x0, y1 - y0]
            [ndof_x, ndof_y] = col.ndofs[:]
            
            [_, w_x, _, w_y, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                  nnodes_y = ndof_y)
            
            w_x = w_x.reshape([ndof_x, 1, 1])
            w_y = w_y.reshape([1, ndof_y, 1])
            
            dcoeff = dx * dy / 4.

            # Loop through cells to calculate error
            col_err    = 0.
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    # Cell information to calculate spatially-integrated analytic
                    # solution
                    [th0, th1] = cell.pos[:]
                    dth        = th1 - th0
                    [ndof_th]  = cell.ndofs[:]
                    
                    [_, _, _, _, thb, _] = qd.quad_xyth(nnodes_th = ndof_th)
                    
                    thf = push_forward(th0, th1, thb)
                    
                    # Calculate error
                    uh_cell = proj.cols[col_key].cells[cell_key].vals
                    uh_cell_intg_xy = dcoeff \
                        * np.sum(w_x * w_y * uh_cell, axis = (0, 1))
                    
                    u_cell_intg_xy = anl_sol_intg_xy(x0, x1, y0, y1, thf)
                    
                    cell_err = np.amax(np.abs(u_cell_intg_xy - uh_cell_intg_xy) / (dx * dy))
                    col_err  = max(col_err, cell_err)
                    
                    u_intg_xy_max = max(u_intg_xy_max, np.amax(np.abs(u_cell_intg_xy)))
                    
                    if kwargs['ref_cell']:
                        err_ind.cols[col_key].cells[cell_key].err = cell_err
                        cell_max_err = max(cell_max_err, cell_err)
                                        
            if kwargs['ref_col']:
                err_ind.cols[col_key].err = col_err
                col_max_err = max(col_max_err, col_err)
            
    # Weight errors to be relative, and calculate hp-steering criteria
    if kwargs['ref_col']:
        col_max_err  /= u_intg_xy_max
        col_ref_thrsh = col_ref_tol * col_max_err
        err_ind.col_max_err = col_max_err

    if kwargs['ref_cell']:
        cell_max_err  /= u_intg_xy_max
        cell_ref_thrsh = cell_ref_tol * cell_max_err
        err_ind.cell_max_err = cell_max_err
            
    # Weight to be relative error
    for col_key, col in col_items:
        if col.is_lf:
            if kwargs['ref_col']: # If we're refining columns
                err_ind.cols[col_key].err /= u_intg_xy_max
                if err_ind.cols[col_key].err >= col_ref_thrsh: # Does this one need to be refined?
                    if err_ind.cols[col_key].ref_form == 'hp': # Does the form of refinement need to be chosen?
                        err_ind.cols[col_key].ref_form = hp_steer_col(mesh, proj, col_key)
                else: # Needn't be refined
                    err_ind.cols[col_key].ref_form = None
                        
                
            if kwargs['ref_cell']: # If we're refining cells
                cell_items = sorted(col.cells.items())
                for cell_key, cell in cell_items:
                    if cell.is_lf:
                        err_ind.cols[col_key].cells[cell_key].err /= u_intg_xy_max
                        
                        if err_ind.cols[col_key].cells[cell_key].err >= cell_ref_thrsh: # Does this one need to be refined?
                            if err_ind.cols[col_key].cells[cell_key].ref_form == 'hp': # Does the form of refinement need to be chosen?
                                err_ind.cols[col_key].cells[cell_key].ref_form = \
                                    hp_steer_cell(mesh, proj, col_key, cell_key)
                        else: # Needn't be refined
                            err_ind.cols[col_key].cells[cell_key].ref_form = None

    return err_ind
