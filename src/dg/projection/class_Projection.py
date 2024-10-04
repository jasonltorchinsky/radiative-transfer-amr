# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports

# Relative Imports
from .projection_column import Projection_Column
from .projection_column.projection_cell import Projection_Cell
from .push_pull import push_forward

from ..mesh import Mesh
from ..quadrature import quad_xyth

class Projection():
    def __init__(self, mesh: Mesh, func = None):
        self.Ls: list     = mesh.Ls
        self.pbcs: list   = mesh.pbcs
        self.has_th: bool = mesh.has_th

        self.mesh: Mesh = mesh # This could be made obselete

        # Write function to have three arguments even if it only has two
        if func is None:
            def f(x, y, th):
                return 0
        else:
            if not mesh.has_th:
                def f(x, y, th):
                    return func(x, y)
            else:
                def f(x, y, th):
                    return func(x, y, th)
        
        # Fill in the Projection
        projection_cols: dict = {}
        col_items: list = sorted(mesh.cols.items())

        for col_key, col in col_items:
            assert(col.is_lf)
                
            [x0, y0, x1, y1] = col.pos
            [ndof_x, ndof_y] = col.ndofs

            [xxb, _, yyb, _, _, _] = quad_xyth(nnodes_x = ndof_x,
                                               nnodes_y = ndof_y)

            xxf: np.ndarray = push_forward(x0, x1, xxb)
            yyf: np.ndarray = push_forward(y0, y1, yyb)

            projection_cells: dict = {}
            cell_items: list = sorted(col.cells.items())

            for cell_key, cell in cell_items:
                assert(cell.is_lf)

                if self.has_th:
                    [th0, th1] = cell.pos
                    [ndof_th]  = cell.ndofs
                    [_, _, _, _, thb, _] = quad_xyth(nnodes_th = ndof_th)
                    
                    thf: np.ndarray = push_forward(th0, th1, thb)
                else:
                    ndof_th: int = 1
                    thf: np.ndarray = np.zeros([1])

                vals: np.ndarray = np.zeros([ndof_x, ndof_y, ndof_th])
                for ii in range(0, ndof_x):
                    x_i: float = xxf[ii]
                    for jj in range(0, ndof_y):
                        y_j: float = yyf[jj]
                        for aa in range(0, ndof_th):
                            th_a: float = thf[aa]
                            vals[ii, jj, aa] = f(x_i, y_j, th_a)
                projection_cells[cell_key] = Projection_Cell(cell, vals)

            projection_cols[col_key] = Projection_Column(col, projection_cells)
        
        self.cols = projection_cols # Columns in mesh
        
    from .cell_intg_xy import cell_intg_xy
    from .col_intg_th import col_intg_th
    from .to_vector import to_vector
    
    def __str__(self):
        msg: str = ( "Ls     :  {}\n".format(self.Ls) +
                     "pbcs   :  {}\n".format(self.pbcs) +
                     "has_th :  {}\n".format(self.has_th) +
                     "cols   :  {}\n".format(sorted(self.cols.keys())))
        
        return msg