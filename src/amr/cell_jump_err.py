import numpy as np

from .Error_Indicator import Error_Indicator
from .hp_steer import hp_steer_cell

import dg.quadrature as qd
from dg.projection import push_forward, pull_back

def cell_jump_err(mesh, proj):

    col_items = sorted(mesh.cols.items())
    ncols = len(col_items)

    # Array to store column jump errors
    err_ind = Error_Indicator(mesh, by_col = False, by_cell = True)
    cell_intg_xys = {}
    
    # Begin by integrating each column with respect to theta
    for col_key, col in col_items:
        if col.is_lf:
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    cell_intg_xys[(col_key, cell_key)] = \
                        intg_cell_bdry_xy(mesh, proj, col_key, cell_key)
            
    # Once we have integrated against x, y for all cells, we need to calculate
    # the jumps
    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, xf, yf] = col.pos[:]
            [dx, dy]         = [xf - x0, yf - y0]
            [ndof_x, ndof_y] = col.ndofs[:]
            
            cell_items = sorted(col.cells.items())
            for cell_key_0, cell_0 in cell_items:
                if cell_0.is_lf:
                    cell_err = 0.
                    
                    for F in range(0, 2):
                        cell_key_1 = cell_0.nhbr_keys[F]
                        
                        cell_intg_xy_0 = cell_intg_xys[(col_key, cell_key_0)][F]
                        cell_intg_xy_1 = cell_intg_xys[(col_key, cell_key_1)][(F+1)%2]
                        
                        cell_err += (cell_intg_xy_0 - cell_intg_xy_1)**2
                        
                    dA = dx * dy
                    cell_err = np.sqrt((1. / dA) * cell_err)
                    
                    err_ind.cols[col_key].cells[cell_key_0].err_ind = cell_err
                    err_ind.cols[col_key].cells[cell_key_0].ref_form = hp_steer_cell(mesh, proj, col_key, cell_key_0)
                    
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
