# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports

# Relative Imports
from .calc_key import calc_key

class Cell:
    def __init__(self, pos: list, idx: int, lv: int, is_lf: bool, ndofs: list,
                 quadrant: int, nhbr_keys: list):
        self.pos: list   = pos    # Position in angular dimension
        self.idx: int    = idx    # Angular index of cell
        self.lv: int     = lv     # Level of angular refinement
        self.key: int    = calc_key(idx, lv) # Unique key for cell
        self.is_lf: bool = is_lf  # Whether cell is a leaf or not
        self.ndofs: list = ndofs  # Degrees of freedom in theta-.
        self.quadrant: int   = quadrant   # Which angular quadrant the cell is in.
        self.nhbr_keys: list = nhbr_keys # Keys for neighboring cells in column.

    def __eq__(self, other): # Doesn't check if cells are in different columns!
        pos: bool = (self.pos == other.pos)
        idx: bool = (self.idx == other.idx)
        lv: bool  = (self.lv  == other.lv)
        key: bool = (self.key == other.key)
        is_lf: bool = (self.is_lf == other.is_lf)
        ndofs: bool = (self.ndofs == other.ndofs)
        quadrant: bool  = (self.quadrant  == other.quadrant)
        nhbr_keys: bool = (self.nhbr_keys == other.nhbr_keys)
        
        return (pos and idx and lv and key and is_lf and ndofs and quadrant
                and nhbr_keys)
        
    def __str__(self):
        pos_str: str = ( "[{:3.2f} pi".format(self.pos[0] / np.pi) +
                    ", {:3.2f} pi]".format(self.pos[1] / np.pi)
                   )
        msg: str = ( "   Cell  :  {}, {}\n".format(self.idx, self.lv) +
                     "    key  :  {}\n".format(self.key) +
                     "    pos  :  {}\n".format(pos_str) +
                     "    is_lf:  {}\n".format(self.is_lf) +
                     "    ndofs:  {}\n".format(self.ndofs) +
                     " quadrant:  {}\n".format(self.quadrant) +
                     "nhbr_keys:  {}\n".format(self.nhbr_keys)
                     )

        return msg
