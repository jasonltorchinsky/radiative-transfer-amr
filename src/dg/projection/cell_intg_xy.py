# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports

# Relative Imports
from .projection_column import Projection_Column
from .projection_column.projection_cell import Projection_Cell
from ..quadrature import quad_xyth

def cell_intg_xy(self, col_key: int, cell_key: int) -> np.ndarray:
    proj_col: Projection_Column = self.cols[col_key]
    assert(proj_col.is_leaf)

    [x0, y0, x1, y1]   = proj_col.pos[:]
    [dx, dy]           = [x1 - x0, y1 - y0]
    [ndof_x, ndof_y]   = proj_col.ndofs[:]
    
    dcoeff: float = (dx * dy) / 4.
    
    [_, wx, _, wy, _, _] = quad_xyth(nnodes_x = ndof_x, nnodes_y = ndof_y)
    wx: np.ndarray = wx.reshape([ndof_x, 1])
    wy: np.ndarray = wy.reshape([1, ndof_y])
    
    proj_cell: Projection_Cell = proj_col.cells[cell_key]
    assert(proj_cell.is_lf)

    [ndof_th]  = proj_cell.ndofs
    
    cell_intg_xy = np.zeros([ndof_th])
    for aa in range(0, ndof_th):
        cell_intg_xy[aa] = dcoeff * np.sum(wx * wy * proj_cell.vals[:, :, aa])
    
    return cell_intg_xy
