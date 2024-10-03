# Standard Library Imports

# Third-Party Library Imports
import json

# Local Library Imports
import consts

# Relative Imports
from .column import Column
from .column.cell import Cell

class Mesh:
    def __init__(self, Ls: list, pbcs: list, ndofs: list = [2, 2, 2], 
                 has_th: bool = False):
        self.Ls: list     = Ls   # Lengths of spatial domain
        self.pbcs: list   = pbcs # Periodicity of spatial domain
        self.has_th: bool = has_th # Include angular domain in mesh?

        # Create first cell
        if has_th:
            cell: Cell = Cell(pos       = [0, 2 * consts.PI],
                              idx       = 0,
                              lv        = 0,
                              is_lf     = True,
                              ndofs     = [ndofs[2]],
                              quadrant  = None,
                              nhbr_keys = [0, 0])
        else:
            cell: Cell = Cell(pos       = [0, 0],
                              idx       = 0,
                              lv        = 0,
                              is_lf     = True,
                              ndofs     = [1],
                              quadrant  = None,
                              nhbr_keys = [None, None])

        # Create first column
        # Determine neighbor keys flags for column
        # F  => Right face, proceed counterclockwise
        nhbr_keys: list = [[None, None], # F = 0
                           [None, None], # F = 1
                           [None, None], # F = 2
                           [None, None]] # F = 3
        
        # Which faces have a neighbor?
        if pbcs[0]: # Periodic in x, is own neighbor in x.
            nhbr_keys[0] = [0, 0]
            nhbr_keys[2] = [0, 0]

        if pbcs[1]: # Periodic in y, is own neighbor in y.
            nhbr_keys[1] = [0, 0]
            nhbr_keys[3] = [0, 0]
            
        col: Column = Column(pos       = [0, 0, Ls[0], Ls[1]],
                             idx       = [0, 0],
                             lv        = 0,
                             is_lf     = True,
                             ndofs     = ndofs[0:2],
                             cells     = {0 : cell},
                             nhbr_keys = nhbr_keys)
        self.cols: dict = {col.key : col} # Columns in mesh
    
    from .to_file import to_file
    from .add_col import add_col
    from .del_col import del_col
    from .ref_col import ref_col
    from .nhbr_cols import nhbr_cols
    from .nhbr_cells_in_nhbr_col import nhbr_cells_in_nhbr_col
    from .nhbr_cells_spt import nhbr_cells_spt
    from .ref_col_spt import ref_col_spt
    from .ref_col_ang import ref_col_ang
    from .ref_cell import ref_cell
    from .ref_mesh import ref_mesh
    from .get_ndof import get_ndof

    def __eq__(self, other):
        Ls: bool = (self.Ls == other.Ls)
        pbcs: bool = (self.pbcs == other.pbcs)
        has_th: bool = (self.has_th == other.has_th)
        cols: bool = (self.cols == other.cols)

        return (Ls and pbcs and has_th and cols)

    def __str__(self):
        msg: str = ( "Ls     :  {}\n".format(self.Ls) +
                     "pbcs   :  {}\n".format(self.pbcs) +
                     "has_th :  {}\n".format(self.has_th) +
                     "cols   :  {}\n".format(sorted(self.cols.keys())) )
        
        return msg