# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports

# Relative Imports
from .projection_column import Projection_Column
from .projection_column.projection_cell import Projection_Cell
from ..quadrature import quad_xyth  

def col_intg_th(self, col_key: int) -> np.ndarray:
    proj_col: Projection_Column = self.cols[col_key]
    assert(proj_col.is_lf)

    [ndof_x, ndof_y] = proj_col.ndofs[:]
    col_intg_th: np.ndarray = np.zeros([ndof_x, ndof_y])
    
    proj_cell_items = sorted(proj_col.cells.items())
    
    for _, proj_cell in proj_cell_items:
        assert(proj_cell.is_lf)
            
        [th0, thf] = proj_cell.pos[:]
        dth: float = thf - th0
        [ndof_th]  = proj_cell.ndofs
        
        [_, _, _, _, _, w_th] = quad_xyth(nnodes_th = ndof_th)
        dcoeff: float = dth / 2.
        
        for ii in range(0, ndof_x):
            for jj in range(0, ndof_y):
                col_intg_th[ii, jj] += \
                        dcoeff * np.sum(w_th * proj_cell.vals[ii, jj, :])
                            
    return col_intg_th