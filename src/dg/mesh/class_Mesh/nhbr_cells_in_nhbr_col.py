# Standard Library Imports
from math import floor

# Third-Party Library Imports

# Local Library Imports
from class_Column import Column

from class_Column.class_Cell import Cell
from class_Column.class_Cell import calc_key as calc_cell_key

# Relative Imports

def nhbr_cells_in_nhbr_col(self, col_key: int, cell_key: int, 
                          nhbr_col_key: int) -> list:

    cell: Cell = self.cols[col_key].cells[cell_key]
    idx: int = cell.idx
    lv: int  = cell.lv
    key: int = cell.key

    nhbr_col: Column = self.cols[nhbr_col_key]
    
    nhbr_cell_keys: list = [None, None]

    # Since the cells in each column are indexed the same, it comes down
    # to checking if the parent, same-level, or child cells are in the
    # neighboring column
    try: # Same-level neighbor
        nhbr_key: int = key
        nhbr: Cell = nhbr_col.cells[nhbr_key]
        if nhbr.is_lf:
            nhbr_cell_keys[0] = nhbr_key
    except:
        None
        
        try: # Parent-level neighbor
            prnt_idx: int = int(idx/2)
            prnt_lv: int  = lv - 1
            nhbr_key: int = calc_cell_key(prnt_idx, prnt_lv)
            nhbr: Cell = nhbr_col.cells[nhbr_key]
            if nhbr.is_lf:
                nhbr_cell_keys[0] = nhbr_key
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
                nhbr_cell_keys[0] = nhbr_0_key
                
            nhbr_1: Cell = nhbr_col.cells[nhbr_1_key]
            if nhbr_1.is_lf:
                nhbr_cell_keys[1] = nhbr_1_key

        except:
            None

    return nhbr_cell_keys
