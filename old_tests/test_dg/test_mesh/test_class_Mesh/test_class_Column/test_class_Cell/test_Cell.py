# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports
from class_Mesh.class_Column.class_Cell import Cell

def test_Cell():
    pos: list = [0., 2. * consts.PI]
    idx: int  = 0
    lv: int   = 0
    is_lf: bool = True
    ndofs: list = [3]
    quadrant: int = None
    nhbr_keys: list = [None, None]

    cell: Cell = Cell(pos, idx, lv, is_lf, ndofs, quadrant, nhbr_keys)