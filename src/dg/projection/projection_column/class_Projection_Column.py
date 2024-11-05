# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports
from dg.mesh.column import Column

# Relative Imports

class Projection_Column:
    def __init__(self, col: Column, projection_cells: dict):
        self.pos: list   = col.pos[:]    # Spatial corners of columns
        self.idx: list   = col.idx[:]    # Spatial index for column
        self.lv: int     = col.lv        # Level of spatial refinement for column
        self.key: int    = col.key       # Unique key for column
        self.is_lf: bool = col.is_lf     # Whether cell is a leaf or not
        self.ndofs: list = col.ndofs[:]  # Degrees of freedom in x-, y-
        self.nhbr_keys: list = col.nhbr_keys[:]  # List of neighbor keys

        self.cells: dict = projection_cells  # Dict of cells in the column

    def __eq__(self, other):
        pos: bool = (self.pos == other.pos)
        idx: bool = (self.idx == other.idx)
        lv: bool  = (self.lv  == other.lv)
        key: bool = (self.key == other.key)
        is_lf: bool = (self.is_lf == other.is_lf)
        ndofs: bool = (self.ndofs == other.ndofs)
        cells: bool = (self.cells == other.cells)
        nhbr_keys: bool = (self.nhbr_keys == other.nhbr_keys)
        
        return (pos and idx and lv and key and is_lf and ndofs and cells
                and nhbr_keys)

    def __str__(self):
        msg: str = ( "    Column:  {}, {}\n".format(self.idx, self.lv) +
                     "       key:  {}\n".format(self.key) +
                     "       pos:  {}\n".format(self.pos) +
                     "     ndofs:  {}\n".format(self.ndofs) +
                     "     cells:  {}\n".format(list(self.cells.keys())) )

        return msg
