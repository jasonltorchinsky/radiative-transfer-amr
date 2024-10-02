import numpy as np

import dg.quadrature as qd    

def intg_cell_xy(mesh, proj, col_key, cell_key):
    """
    Integrate a column with respect to theta.
    """

    col = mesh.cols[col_key]
    if col.is_lf:
        [x0, y0, x1, y1]   = col.pos[:]
        [dx, dy]           = [x1 - x0, y1 - y0]
        [ndof_x, ndof_y]   = col.ndofs[:]
        
        dcoeff = (dx * dy) / 4.
        
        [_, wx, _, wy, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                nnodes_y = ndof_y)
        wx  = wx.reshape([ndof_x, 1])
        wy  = wy.reshape([1, ndof_y])
        
        cell = col.cells[cell_key]
        if cell.is_lf:
            [ndof_th]  = cell.ndofs
            
            cell_intg_xy = np.zeros([ndof_th])
                
            proj_cell = proj.cols[col_key].cells[cell_key]

            for aa in range(0, ndof_th):
                cell_intg_xy[aa] = \
                    dcoeff * np.sum(wx * wy * proj_cell.vals[:, :, aa])
            
            return cell_intg_xy.reshape([ndof_th])

    else:
        return None
