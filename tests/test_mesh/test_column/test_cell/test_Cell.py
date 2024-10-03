# Standard Library Imports
import os

# Third-Party Library Imports

# Local Library Imports
import consts
from mesh.column.cell import Cell

def test_Cell(tmp_path):
    # Create three cells
    cell_pos: list = [0., 2. * consts.PI]
    cell_idx: int  = 0
    cell_lv: int   = 0
    cell_is_lf: bool = True
    cell_ndofs: list = [3]
    cell_quadrant: int = None
    cell_nhbr_keys: list = [None, None]

    cell: Cell = Cell(cell_pos, cell_idx, cell_lv, cell_is_lf, cell_ndofs,
                      cell_quadrant, cell_nhbr_keys)
    same_cell: Cell = Cell(cell_pos, cell_idx, cell_lv, cell_is_lf, cell_ndofs,
                           cell_quadrant, cell_nhbr_keys)
    diff_cell: Cell = Cell([0., consts.PI], cell_idx, cell_lv, cell_is_lf, 
                           cell_ndofs, cell_quadrant, cell_nhbr_keys)

    assert(cell == same_cell)
    assert(cell != diff_cell)

    file_name: str = "cell_str.txt"
    file_path: str = os.path.join(tmp_path, file_name)
    file = open(file_path, "w")
    file.write(cell.__str__())
    file.close()