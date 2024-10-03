# Standard Library Imports
from math import floor

# Third-Party Library Imports

# Local Library Imports
from .cell import Cell
from .cell import calc_key as calc_cell_key

# Relative Imports

def nhbr_cell(self, cell: Cell, nhbr_loc: str = "+") -> Cell:
    # Get what the idx, lv of the neighbor would be
    idx: int = cell.idx
    lv: int  = cell.lv
    
    nhbr_idx: int = idx
    if nhbr_loc == "+":
        nhbr_idx += 1
    elif nhbr_loc == "-":
        nhbr_idx -= 1

    nhbr_idx: int = nhbr_idx % (2 ** lv)

    nhbr_key: int = calc_cell_key(nhbr_idx, lv)
    # If neighbor is a leaf, we"re done
    try: # Instead of searching through keys of mesh.is_lf, we just go for it
        nhbr: Cell = self.cells[nhbr_key]
        if nhbr.is_lf:
            return nhbr
    except:
        None

    # Neighbor is not leaf. Check if neighbor"s parent is leaf
    nhbr_prnt_idx: int = floor(nhbr_idx / 2)
    nhbr_prnt_key: int = calc_cell_key(nhbr_prnt_idx, lv - 1)
    # If neighbor"s parent is a leaf, we"re done
    try: # Instead of searching through keys of mesh.is_lf, we just go for it
        nhbr_prnt: Cell = self.cells[nhbr_prnt_key]
        if nhbr_prnt.is_lf:
            return nhbr_prnt
    except:
        None

    # Neighbor"s parent is not a leaf. Neighbor"s children must be leaves
    if nhbr_loc == "+":
        nhbr_idx: int = 2 * nhbr_idx
    elif nhbr_loc == "-":
        nhbr_idx: int = 2 * nhbr_idx + 1

    nhbr_key: int = calc_cell_key(nhbr_idx, lv + 1)
    # If neighbor"s children are leaves, we"re done
    try: # Instead of searching through keys of mesh.is_lf, we just go for it
        nhbr: Cell = self.cells[nhbr_key]
        if nhbr.is_lf:
            return nhbr
    except:
        None

    # Otherwise, there"s no neighbor (BUT THERE SHOULD ALWAYS BE ONE)
    return None