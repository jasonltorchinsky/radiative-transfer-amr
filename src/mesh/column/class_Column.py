# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports

# Relative Imports
from .calc_key import calc_key

class Column:
    def __init__(self, pos: list, idx: list, lv: int, is_lf: bool, ndofs: list, 
                 cells: dict, nhbr_keys: list):
        self.pos: list   = pos   # Spatial corners of columns
        self.idx: list   = idx   # Spatial index for column
        self.lv: int     = lv    # Level of spatial refinement for column
        self.key: int    = calc_key(idx, lv) # Unique key for column
        self.is_lf: bool = is_lf # Whether column is a leaf or not
        self.ndofs: list = ndofs # Degrees of freedom in x-, y-.
        self.cells: dict = cells # Dict of cells in the column
        self.nhbr_keys: list = nhbr_keys  # List of neighbor keys

    from .add_cell import add_cell
    from .del_cell import del_cell
    from .nhbr_cell import nhbr_cell

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
                     "     is_lf:  {}\n".format(self.is_lf) +
                     "     ndofs:  {}\n".format(self.ndofs) +
                     "     cells:  {}\n".format(list(self.cells.keys())) +
                     " nhbr_keys:  {}\n".format(self.nhbr_keys)
                   )

        return msg