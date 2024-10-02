# Standard Library Imports
from math import floor

# Third-Party Library Imports

# Local Library Imports
from class_Column import Column
from class_Column import calc_key as calc_col_key

from class_Column.class_Cell import Cell
from class_Column.class_Cell import calc_key as calc_cell_key

# Relative Imports

def nhbr_cell_ang(col, cell, nhbr_loc = "+"):

    # Get what the idx, lv of the neighbor would be
    idx = cell.idx
    lv = cell.lv
    
    nhbr_idx = idx
    if nhbr_loc == "+":
        nhbr_idx += 1
    elif nhbr_loc == "-":
        nhbr_idx -= 1

    nhbr_idx = nhbr_idx % (2 ** lv)

    nhbr_key = calc_cell_key(nhbr_idx, lv)
    # If neighbor is a leaf, we"re done
    try: # Instead of searching through keys of mesh.is_lf, we just go for it
        nhbr = col.cells[nhbr_key]
        if nhbr.is_lf:
            return nhbr
    except:
        None

    # Neighbor is not leaf. Check if neighbor"s parent is leaf
    nhbr_prnt_idx = floor(nhbr_idx / 2)
    nhbr_prnt_key = calc_cell_key(nhbr_prnt_idx, lv - 1)
    # If neighbor"s parent is a leaf, we"re done
    try: # Instead of searching through keys of mesh.is_lf, we just go for it
        nhbr_prnt = col.cells[nhbr_prnt_key]
        if nhbr_prnt.is_lf:
            
            return nhbr_prnt
    except:
        None

    # Neighbor"s parent is not a leaf. Neighbor"s children must be leaves
    if nhbr_loc == "+":
        nhbr_idx = 2 * nhbr_idx
    elif nhbr_loc == "-":
        nhbr_idx = 2 * nhbr_idx + 1

    nhbr_key = calc_cell_key(nhbr_idx, lv + 1)
    # If neighbor"s children are leaves, we"re done
    try: # Instead of searching through keys of mesh.is_lf, we just go for it
        nhbr = col.cells[nhbr_key]
        if nhbr.is_lf:
            
            return nhbr
    except:
        None
    

    # Otherwise, there"s no neighbor (BUT THERE SHOULD ALWAYS BE ONE)
    return None

def get_cell_spt_nhbr(mesh, col, cell, axis = 0, nhbr_loc = "+"):

    [flag, nhbr_col_1, nhbr_col_2] = get_col_nhbr(mesh, col, axis, nhbr_loc)

    nhbr_cells = [[nhbr_col_1, [None, None]], [nhbr_col_2, [None, None]]]
    
    if flag == "nn":
        # No neighboring columns
        return nhbr_cells
    
    idx = cell.idx
    lv = cell.lv
    key = cell.key

    for nn in range(0, 2):
        nhbr_col = [nhbr_col_1, nhbr_col_2][nn]
        # Since the cells in each column are indexed the same, it comes down
        # to checking if the parent, same-level, or child cells are in the
        # neighboring column
        try: # Same-level neighbor
            nhbr_key = key
            nhbr = nhbr_col.cells[nhbr_key]
            if nhbr.is_lf:
                nhbr_cells[nn][1][0] = nhbr
        except:
            None

        try: # Parent-level neighbor
            prnt_idx = int(idx/2)
            prnt_lv = lv - 1
            nhbr_key = calc_cell_key(prnt_idx, prnt_lv)
            nhbr = nhbr_col.cells[nhbr_key]
            if nhbr.is_lf:
                nhbr_cells[nn][1][0] = nhbr
        except:
            None

        try: # Child-level neighbor
            chld_0_idx = 2*idx
            chld_1_idx = 2*idx + 1
            chld_lv = lv + 1
            nhbr_0_key = calc_cell_key(chld_0_idx, chld_lv)
            nhbr_1_key = calc_cell_key(chld_1_idx, chld_lv)

            nhbr_0 = nhbr_col.cells[nhbr_0_key]
            if nhbr_0.is_lf:
                nhbr_cells[nn][1][0] = nhbr_0

            nhbr_1 = nhbr_col.cells[nhbr_1_key]
            if nhbr_1.is_lf:
                nhbr_cells[nn][1][1] = nhbr_1

        except:
            None

    return nhbr_cells

def get_cell_nhbr(col, cell, nhbr_loc = "+"):
    
    return get_cell_ang_nhbr(col, cell, nhbr_loc)

def get_cell_nhbr_in_col(mesh, col_key, cell_key, nhbr_col_key):

    cell     = mesh.cols[col_key].cells[cell_key]
    nhbr_col = mesh.cols[nhbr_col_key]
    
    nhbr_cell_keys = [None, None]
    
    idx = cell.idx
    lv  = cell.lv
    key = cell.key

    # Since the cells in each column are indexed the same, it comes down
    # to checking if the parent, same-level, or child cells are in the
    # neighboring column
    try: # Same-level neighbor
        nhbr_key = key
        nhbr = nhbr_col.cells[nhbr_key]
        if nhbr.is_lf:
            nhbr_cell_keys[0] = nhbr_key
    except:
        None
        
        try: # Parent-level neighbor
            prnt_idx = int(idx/2)
            prnt_lv = lv - 1
            nhbr_key = calc_cell_key(prnt_idx, prnt_lv)
            nhbr = nhbr_col.cells[nhbr_key]
            if nhbr.is_lf:
                nhbr_cell_keys[0] = nhbr_key
        except:
            None
            
        try: # Child-level neighbor
            chld_0_idx = 2*idx
            chld_1_idx = 2*idx + 1
            chld_lv    = lv + 1
            nhbr_0_key = calc_cell_key(chld_0_idx, chld_lv)
            nhbr_1_key = calc_cell_key(chld_1_idx, chld_lv)
            
            nhbr_0 = nhbr_col.cells[nhbr_0_key]
            if nhbr_0.is_lf:
                nhbr_cell_keys[0] = nhbr_0_key
                
                nhbr_1 = nhbr_col.cells[nhbr_1_key]
            if nhbr_1.is_lf:
                nhbr_cell_keys[1] = nhbr_1_key

        except:
            None

    return nhbr_cell_keys
