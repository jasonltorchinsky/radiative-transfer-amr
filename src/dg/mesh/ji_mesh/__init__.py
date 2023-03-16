import numpy as np

from .Column import Column
from .Cell import Cell

from .get_nhbr import get_cell_nhbr_in_col

class Mesh:
    def __init__(self, Ls, pbcs, ndofs = [2, 2, 2], has_th = False):
        self.Ls     = Ls     # Lengths of spatial domain
        self.pbcs   = pbcs   # Periodicity of spatial domain
        self.has_th = has_th # Include angular domain in mesh?

        # Create first cell
        if has_th:
            cell = Cell(pos       = [0, 2 * np.pi],
                        idx       = 0,
                        lv        = 0,
                        is_lf     = True,
                        ndofs     = [ndofs[2]],
                        quad      = None,
                        nhbr_keys = [0, 0])
        else:
            cell = Cell(pos       = [0, 0],
                        idx       = 0,
                        lv        = 0,
                        is_lf     = True,
                        ndofs     = [0],
                        quad      = None,
                        nhbr_keys = [None, None])

        # Create first column.
        # Determine neighbor keys flags for column
        # F  => Right face, proceed counterclockwise
        nhbr_keys = [
            [None, None], # F = 0
            [None, None], # F = 1
            [None, None], # F = 2
            [None, None]  # F = 3
        ]
        
        # Which faces have a neighbor?
        if pbcs[0]: # Periodic in x, is own neighbor in x.
            nhbr_keys[0] = [0, None]
            nhbr_keys[2] = [0, None]

        if pbcs[1]: # Periodic in y, is own neighbor in y.
            nhbr_keys[1] = [0, None]
            nhbr_keys[3] = [0, None]
            
        col = Column(pos       = [0, 0, Ls[0], Ls[1]],
                     idx       = [0, 0],
                     lv        = 0,
                     is_lf     = True,
                     ndofs     = ndofs[0:2],
                     cells     = {0 : cell},
                     nhbr_keys = nhbr_keys)
        self.cols = {0 :  col} # Columns in mesh
        
    def __str__(self):
        msg = ( 'Ls     :  {}\n'.format(self.Ls) +
                'pbcs   :  {}\n'.format(self.pbcs) +
                'has_th :  {}\n'.format(self.has_th) +
                'cols   :  {}\n'.format(sorted(self.cols.keys()))
               )
        
        return msg

    from .add_col import add_col
    from .del_col import del_col
    from .ref_col import ref_col
    from .ref_col_spt import ref_col_spt
    from .ref_col_ang import ref_col_ang
    from .ref_cell import ref_cell
    from .ref_mesh import ref_mesh
