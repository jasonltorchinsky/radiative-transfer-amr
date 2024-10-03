# Standard Library Imports
import os

# Third-Party Library Imports

# Local Library Imports
import consts
from dg.mesh.column.cell import Cell
from dg.mesh.column import Column

def test_Column(tmp_path):
    # Create two cells
    cell_pos: list = [0., 2. * consts.PI]
    cell_idx: int  = 0
    cell_lv: int   = 0
    cell_is_lf: bool = True
    cell_ndofs: list = [3]
    cell_quadrant: int = None
    cell_nhbr_keys: list = [None, None]

    cell: Cell = Cell(cell_pos, cell_idx, cell_lv, cell_is_lf, cell_ndofs,
                      cell_quadrant, cell_nhbr_keys)
    diff_cell: Cell = Cell([0., consts.PI], cell_idx, cell_lv, cell_is_lf, cell_ndofs,
                           cell_quadrant, cell_nhbr_keys)
    

    # Create three columns
    col_pos: list = [[0., 1.], [-0.5, 0.5]]
    col_idx: int  = [0, 0]
    col_lv: int   = 0
    col_is_lf: bool = True
    col_ndofs: list = [3, 4]
    cells: dict = {cell.key : cell}
    # F  => Right face, proceed counterclockwise
    col_nhbr_keys: list = [[None, None], # F = 0
                           [None, None], # F = 1
                           [None, None], # F = 2
                           [None, None]] # F = 3
    
    col: Column = Column(col_pos, col_idx, col_lv, col_is_lf, col_ndofs, cells,
                         col_nhbr_keys)
    same_col: Column = Column(col_pos, col_idx, col_lv, col_is_lf, col_ndofs,
                              cells, col_nhbr_keys)
    
    diff_cells: dict = {diff_cell.key : diff_cell}
    diff_col: Column = Column(col_pos, col_idx, col_lv, col_is_lf, col_ndofs,
                              diff_cells, col_nhbr_keys)
    
    assert(col == same_col)
    assert(col != diff_col)

    file_name: str = "col_str.txt"
    file_path: str = os.path.join(tmp_path, file_name)
    file = open(file_path, "w")
    file.write(col.__str__())
    file.close()