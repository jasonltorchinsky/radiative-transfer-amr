import numpy as np

from .Error_Indicator import Error_Indicator
from .hp_steer import hp_steer_cell

import dg.quadrature as qd
from dg.projection import push_forward, pull_back

def cell_jump_err(mesh, proj, **kwargs):

    default_kwargs = {'ref_col'      : False,
                      'col_ref_form' : None,
                      'col_ref_kind' : None,
                      'col_ref_tol'  : None,
                      'ref_cell'      : True,
                      'cell_ref_form' : 'hp',
                      'cell_ref_kind' : 'ang',
                      'cell_ref_tol'  : 0.85}
    kwargs = {**default_kwargs, **kwargs}
    
    err_ind = Error_Indicator(mesh, **kwargs)
    
    col_items = sorted(mesh.cols.items())
    
    # Track maximum error(s) to calculate hp-steering only where needed
    cell_max_err  = 0.
    cell_ref_tol  = kwargs['cell_ref_tol']
    
    # Begin by integrating each column with respect to theta
    cell_intg_xys = {}
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    cell_intg_xys[(col_key, cell_key)] = \
                        intg_cell_bdry_xy(mesh, proj, col_key, cell_key)
            
    # Once we have spatially-integrated alogn angular faces of each cell, we
    # calculate the jumps
    for col_key, col in col_items:
        if col.is_lf:
            # Column information for weighting
            [x0, y0, xf, yf] = col.pos[:]
            [dx, dy]         = [xf - x0, yf - y0]
            [ndof_x, ndof_y] = col.ndofs[:]

            dA = dx * dy

            #Loop through cells to calculate error
            cell_items = sorted(col.cells.items())
            for cell_key_0, cell_0 in cell_items:
                if cell_0.is_lf:
                    cell_err = 0.
                    
                    for F in range(0, 2):
                        cell_key_1 = cell_0.nhbr_keys[F]
                        
                        cell_intg_xy_0 = cell_intg_xys[(col_key, cell_key_0)][F]
                        cell_intg_xy_1 = cell_intg_xys[(col_key, cell_key_1)][(F+1)%2]
                        
                        cell_err += (cell_intg_xy_0 - cell_intg_xy_1)**2
                        
                    cell_err = np.sqrt((1. / dA) * cell_err)
                    cell_max_err = max(cell_max_err, cell_err)
                    if kwargs['ref_cell']:
                        err_ind.cols[col_key].cells[cell_key_0].err = cell_err
                        
    if kwargs['ref_cell']:
        err_ind.cell_max_err = cell_max_err
        cell_ref_thrsh = cell_ref_tol * cell_max_err
        
        for col_key, col in col_items:
            if col.is_lf:
                cell_items = sorted(col.cells.items())
                for cell_key, cell in cell_items:
                    if err_ind.cols[col_key].cells[cell_key].err >= cell_ref_thrsh:
                        if err_ind.cols[col_key].cells[cell_key].ref_form == 'hp':
                            err_ind.cols[col_key].cells[cell_key].ref_form = hp_steer_cell(mesh, proj, col_key, cell_key)
                    else:
                        err_ind.cols[col_key].cells[cell_key].ref_form = None
                    
    return err_ind

def intg_cell_bdry_xy(mesh, proj, col_key, cell_key):
    '''
    Integrate the angular-top and -bot in space.
    '''
    col = mesh.cols[col_key]
    if col.is_lf:
        [x0, y0, xf, yf] = col.pos[:]
        [dx, dy]         = [xf - x0, yf - y0]
        [ndof_x, ndof_y] = col.ndofs[:]
        
        [_, w_x, _, w_y, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                              nnodes_y = ndof_y)

        w_x = w_x.reshape(ndof_x, 1)
        w_y = w_y.reshape(1, ndof_y)
        
        dcoeff = (dx * dy / 4.)
        
        proj_col  = proj.cols[col_key]
        
        cell = col.cells[cell_key]
        if cell.is_lf:
            [ndof_th]  = cell.ndofs
            
            proj_cell  = proj_col.cells[cell_key]
            uh_cell = proj_cell.vals
        
            # Store spatial integral along each angular face
            # F = 0 => Bottom
            cell_intg_xy = [0, 0]
            
            for F in range(0, 2):
                if F == 0:
                    th_idx = 0
                else:
                    th_idx = ndof_th - 1
                    
                cell_intg_xy[F] = np.sum(dcoeff * w_x * w_y * uh_cell[:, :, th_idx],
                                         axis = (0, 1))
                                
        return cell_intg_xy
