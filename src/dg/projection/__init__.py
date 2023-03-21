import numpy as np

from .Projection_Column import Projection_Column
from .Projection_Cell import Projection_Cell

from .push_pull import push_forward, pull_back

from ..quadrature import quad_xyth

class Projection():
    def __init__(self, mesh, func):
        self.Ls     = mesh.Ls
        self.pbcs   = mesh.pbcs   # Periodicity of spatial domain
        self.has_th = mesh.has_th

        # Write function to have three arguments even if it only has two
        if not mesh.has_th:
            def f(x, y, th):
                return func(x, y)
        else:
            def f(x, y, th):
                return func(x, y, th)
            
        proj_cols = {}
            
        col_items = sorted(mesh.cols.items())
        for col_key, col in col_items:
            if col.is_lf:
                [x0, y0, x1, y1] = col.pos
                [dx, dy] = [x1 - x0, y1 - y0]
                [ndof_x, ndof_y] = col.ndofs

                [xxb, _, yyb, _, _, _] = quad_xyth(nnodes_x = ndof_x,
                                                   nnodes_y = ndof_y)

                xxf = push_forward(x0, x1, xxb)
                yyf = push_forward(y0, y1, yyb)

                proj_cells = {}
                
                cell_items = sorted(col.cells.items())
                for cell_key, cell in cell_items:
                    if cell.is_lf:
                        [th0, th1] = cell.pos
                        dth        = th1 - th0
                        [ndof_th]  = cell.ndofs
                        if self.has_th:
                            [_, _, _, _, thb, _] = quad_xyth(nnodes_th = ndof_th)
                            
                            thf = push_forward(th0, th1, thb)
                        else:
                            thf = np.zeros([1])

                        vals = np.zeros([ndof_x, ndof_y, ndof_th])
                        for ii in range(0, ndof_x):
                            x_i = xxf[ii]
                            for jj in range(0, ndof_y):
                                y_j = yyf[jj]
                                for aa in range(0, ndof_th):
                                    th_a = thf[aa]

                                    vals[ii, jj, aa] = f(x_i, y_j, th_a)

                        proj_cells[cell_key] = Projection_Cell(cell, vals)

                proj_cols[col_key] = Projection_Column(col, proj_cells)
        
        self.cols = proj_cols # Columns in mesh
        
    def __str__(self):
        msg = ( 'Ls     :  {}\n'.format(self.Ls) +
                'pbcs   :  {}\n'.format(self.pbcs) +
                'has_th :  {}\n'.format(self.has_th) +
                'cols   :  {}\n'.format(sorted(self.cols.keys()))
               )
        
        return msg

    from .vectorization import to_vector
