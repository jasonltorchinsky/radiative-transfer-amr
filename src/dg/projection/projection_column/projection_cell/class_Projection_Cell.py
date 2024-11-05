# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports
import consts
from dg.mesh.column.cell import Cell

# Relative Imports

class Projection_Cell():
    def __init__(self, cell: Cell, vals: np.ndarray):
        self.pos: list   = cell.pos[:]   # Position in angular dimension
        self.idx: int    = cell.idx      # Angular index of cell
        self.lv: int     = cell.lv       # Level of angular refinement
        self.key: int    = cell.key      # Unique key for cell
        self.is_lf: bool = cell.is_lf    # Whether cell is a leaf or not
        self.ndofs: list = cell.ndofs[:] # Degrees of freedom in theta-..
        self.quadrant: int   = cell.quadrant  # Which angular quadrant the cell is in.
        self.nhbr_keys: list = cell.nhbr_keys # Keys for neighboring cells in column.

        self.vals: np.ndarray = vals

    def __eq__(self, other): # Doesn't check if cells are in different columns!
        pos: bool = (self.pos == other.pos)
        idx: bool = (self.idx == other.idx)
        lv: bool  = (self.lv  == other.lv)
        key: bool = (self.key == other.key)
        is_lf: bool = (self.is_lf == other.is_lf)
        ndofs: bool = (self.ndofs == other.ndofs)
        quadrant: bool  = (self.quadrant  == other.quadrant)
        nhbr_keys: bool = (self.nhbr_keys == other.nhbr_keys)

        vals: bool = np.min(self.vals == other.vals)
        
        return (pos and idx and lv and key and is_lf and ndofs and quadrant
                and nhbr_keys and vals)
        
    def __str__(self):
        pos_str: str = ( "[{:3.2f} pi".format(self.pos[0] / consts.PI) +
                         ", {:3.2f} pi]".format(self.pos[1] / consts.PI) )
        msg: str = ( "   Cell  :  {}, {}\n".format(self.idx, self.lv) +
                     "    key  :  {}\n".format(self.key) +
                     "    pos  :  {}\n".format(pos_str) +
                     "    ndofs:  {}\n".format(self.ndofs) )

        return msg
