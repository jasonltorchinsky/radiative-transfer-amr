import copy
import numpy as np

import dg.quadrature as qd    

def intg_col_th(mesh, proj, col_key):
    '''
    Integrate a column with respect to theta.
    '''

    col = mesh.cols[col_key]
    if col.is_lf:
        [ndof_x, ndof_y] = col.ndofs[:]
        col_intg_th = np.zeros([ndof_x, ndof_y])
        
        cell_items = sorted(col.cells.items())
        
        for cell_key, cell in cell_items:
            if cell.is_lf:
                [th0, thf] = cell.pos[:]
                dth        = thf - th0
                [ndof_th]  = cell.ndofs[:]
                
                [_, _, _, _, _, w_th] = qd.quad_xyth(nnodes_th = ndof_th)
                dcoeff = dth / 2.
                
                proj_cell = proj.cols[col_key].cells[cell_key]
                
                for ii in range(0, ndof_x):
                    for jj in range(0, ndof_y):
                        col_intg_th[ii, jj] += \
                                dcoeff * np.sum(w_th * proj_cell.vals[ii, jj, :])
                                
        return col_intg_th

    else:
        return None
