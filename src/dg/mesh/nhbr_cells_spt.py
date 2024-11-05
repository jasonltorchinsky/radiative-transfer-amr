# Standard Library Imports
from math import floor

# Third-Party Library Imports

# Local Library Imports

# Relative Imports
from .column import Column

from .column.cell import Cell
from .column.cell import calc_key as calc_cell_key

def nhbr_cells_spt(self, col_key: int, cell_key: int, axis: int = 0,
                   nhbr_loc: str = "+") -> list:
    col: Column = self.cols[col_key]
    [flag, nhbr_col_1, nhbr_col_2] = self.nhbr_col(self, col, axis, nhbr_loc)

    nhbr_cells: list = [[nhbr_col_1, [None, None]], [nhbr_col_2, [None, None]]]
    
    if flag == "nn":
        # No neighboring columns
        return nhbr_cells
    
    cell: Cell = col.cells[cell_key]
    idx: int   = cell.idx
    lv: int    = cell.lv
    key: int   = cell.key

    for nn in range(0, 2):
        nhbr_col: Column = [nhbr_col_1, nhbr_col_2][nn]
        # Since the cells in each column are indexed the same, it comes down
        # to checking if the parent, same-level, or child cells are in the
        # neighboring column
        try: # Same-level neighbor
            nhbr_key: int = key
            nhbr: Cell = nhbr_col.cells[nhbr_key]
            if nhbr.is_lf:
                nhbr_cells[nn][1][0] = nhbr
        except:
            None

        try: # Parent-level neighbor
            prnt_idx: int = int(idx/2)
            prnt_lv: int  = lv - 1
            nhbr_key: int = calc_cell_key(prnt_idx, prnt_lv)
            nhbr: Cell = nhbr_col.cells[nhbr_key]
            if nhbr.is_lf:
                nhbr_cells[nn][1][0] = nhbr
        except:
            None

        try: # Child-level neighbor
            chld_0_idx: int = 2*idx
            chld_1_idx: int = 2*idx + 1
            chld_lv: int    = lv + 1
            nhbr_0_key: int = calc_cell_key(chld_0_idx, chld_lv)
            nhbr_1_key: int = calc_cell_key(chld_1_idx, chld_lv)

            nhbr_0: Cell = nhbr_col.cells[nhbr_0_key]
            if nhbr_0.is_lf:
                nhbr_cells[nn][1][0] = nhbr_0

            nhbr_1: Cell = nhbr_col.cells[nhbr_1_key]
            if nhbr_1.is_lf:
                nhbr_cells[nn][1][1] = nhbr_1
        except:
            None

    return nhbr_cells