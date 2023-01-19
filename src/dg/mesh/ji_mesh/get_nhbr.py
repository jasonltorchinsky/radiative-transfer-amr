import sys
from math import floor

from .calc_key import calc_col_key, calc_cell_key


def get_col_nhbr(mesh, col, axis = 0, nhbr_loc = '+'):

    # Get what the idx, lv of the neighbor would be
    idx = col.idx[:]
    lv = col.lv
    
    nhbr_idx = idx[:]
    if nhbr_loc == '+':
        nhbr_idx[axis] += 1
    elif nhbr_loc == '-':
        nhbr_idx[axis] -= 1

    if mesh.pbcs[axis]:
        nhbr_idx[axis] = nhbr_idx[axis] % (2 ** lv)

    # If we look for a neighbor outside of the mesh, we should report no-neighbor
    if (nhbr_idx[axis] < 0) or (nhbr_idx[axis] >= (2 ** lv)):
        flag = 'nn'
        nhbr_1 = None
        nhbr_2 = None
        
        return [flag, nhbr_1, nhbr_2]
    
    nhbr_key = calc_col_key(nhbr_idx, lv)
    # If neighbor is a leaf, we're done
    try: # Instead of searching through keys of mesh.is_lf, we just go for it
        nhbr = mesh.cols[nhbr_key]
        if nhbr.is_lf:
            flag = 'f0'
            nhbr_1 = nhbr
            nhbr_2 = None
            
            return [flag, nhbr_1, nhbr_2]
    except:
        None
    
    # Neighbor is not leaf. Check if neighbor's parent is leaf
    # Only need to check is cell is refined at least twice
    if lv >= 2:
        nhbr_prnt_idx = [floor(nhbr_idx[0] / 2), floor(nhbr_idx[1] / 2)]
        nhbr_prnt_key = calc_col_key(nhbr_prnt_idx, lv - 1)
        # If neighbor's parent is a leaf, we're done
        try: # Instead of searching through keys of mesh.is_lf, we just go for it
            nhbr_prnt = mesh.cols[nhbr_prnt_key]
            if nhbr_prnt.is_lf:
                if (nhbr_idx[axis] == 2 * floor(nhbr_idx[axis] / 2)):
                    flag = 'pm'
                else:
                    flag = 'pp'
                    
                nhbr_1 = nhbr_prnt
                nhbr_2 = None
            
                return [flag, nhbr_1, nhbr_2]
        except:
            None

    # Neighbor's parent is not a leaf. Neighbor's children must be leaves
    if axis == 0:
        if nhbr_loc == '+':
            nhbr_1_idx = [2 * nhbr_idx[0], 2 * nhbr_idx[1]]
            nhbr_2_idx = [2 * nhbr_idx[0], 2 * nhbr_idx[1] + 1]
        elif nhbr_loc == '-':
            nhbr_1_idx = [2 * nhbr_idx[0] + 1, 2 * nhbr_idx[1]]
            nhbr_2_idx = [2 * nhbr_idx[0] + 1, 2 * nhbr_idx[1] + 1]
    elif axis == 1:
        if nhbr_loc == '+':
            nhbr_1_idx = [2 * nhbr_idx[0],    2 * nhbr_idx[1]]
            nhbr_2_idx = [2 * nhbr_idx[0]+ 1, 2 * nhbr_idx[1]]
        elif nhbr_loc == '-':
            nhbr_1_idx = [2 * nhbr_idx[0],     2 * nhbr_idx[1] + 1]
            nhbr_2_idx = [2 * nhbr_idx[0] + 1, 2 * nhbr_idx[1] + 1]

    nhbr_1_key = calc_col_key(nhbr_1_idx, lv + 1)
    nhbr_2_key = calc_col_key(nhbr_2_idx, lv + 1)
    # If neighbor's children are leaves, we're done
    try: # Instead of searching through keys of mesh.is_lf, we just go for it
        nhbr_1 = mesh.cols[nhbr_1_key]
        nhbr_2 = mesh.cols[nhbr_2_key]
        if nhbr_1.is_lf and nhbr_2.is_lf:
            flag = 'cc'
            nhbr_1 = nhbr_1
            nhbr_2 = nhbr_2
            
            return [flag, nhbr_1, nhbr_2]
    except:
        None
    
    # Otherwise, there's no neighbor
    flag = 'nn'
    nhbr_1 = None
    nhbr_2 = None

    return [flag, nhbr_1, nhbr_2]

def get_cell_ang_nhbr(col, cell, nhbr_loc = '+'):

    # Get what the idx, lv of the neighbor would be
    idx = cell.idx
    lv = cell.lv
    
    nhbr_idx = idx
    if nhbr_loc == '+':
        nhbr_idx += 1
    elif nhbr_loc == '-':
        nhbr_idx -= 1

    nhbr_idx = nhbr_idx % (2 ** lv)

    nhbr_key = calc_cell_key(nhbr_idx, lv)
    # If neighbor is a leaf, we're done
    try: # Instead of searching through keys of mesh.is_lf, we just go for it
        nhbr = col.cells[nhbr_key]
        if nhbr.is_lf:
            return nhbr
    except:
        None

    # Neighbor is not leaf. Check if neighbor's parent is leaf
    nhbr_prnt_idx = floor(nhbr_idx / 2)
    nhbr_prnt_key = calc_cell_key(nhbr_prnt_idx, lv - 1)
    # If neighbor's parent is a leaf, we're done
    try: # Instead of searching through keys of mesh.is_lf, we just go for it
        nhbr_prnt = col.cells[nhbr_prnt_key]
        if nhbr_prnt.is_lf:
            
            return nhbr_prnt
    except:
        None

    # Neighbor's parent is not a leaf. Neighbor's children must be leaves
    if nhbr_loc == '+':
        nhbr_idx = 2 * nhbr_idx
    elif nhbr_loc == '-':
        nhbr_idx = 2 * nhbr_idx + 1

    nhbr_key = calc_cell_key(nhbr_idx, lv + 1)
    # If neighbor's children are leaves, we're done
    try: # Instead of searching through keys of mesh.is_lf, we just go for it
        nhbr = col.cells[nhbr_key]
        if nhbr.is_lf:
            
            return nhbr
    except:
        None
    

    # Otherwise, there's no neighbor (BUT THERE SHOULD ALWAYS BE ONE)
    return None

def get_cell_spt_nhbr(mesh, col, cell, axis = 0, nhbr_loc = '+'):

    [flag, nhbr_col_1, nhbr_col_2] = get_col_nhbr(mesh, col, axis, nhbr_loc)

    nhbr_cells = [[nhbr_col_1, [None, None]], [nhbr_col_2, [None, None]]]
    
    if flag == 'nn':
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

    return nhbr_cells

def get_cell_nhbr(col, cell, nhbr_loc = '+'):
    
    return get_cell_ang_nhbr(col, cell, nhbr_loc)

def get_cell_nhbr_in_col(cell, nhbr_col):

    nhbr_cells = [None, None]
    
    idx = cell.idx
    lv = cell.lv
    key = cell.key

    # Since the cells in each column are indexed the same, it comes down
    # to checking if the parent, same-level, or child cells are in the
    # neighboring column
    try: # Same-level neighbor
        nhbr_key = key
        nhbr = nhbr_col.cells[nhbr_key]
        if nhbr.is_lf:
            nhbr_cells[0] = nhbr
    except:
        None
        
        try: # Parent-level neighbor
            prnt_idx = int(idx/2)
            prnt_lv = lv - 1
            nhbr_key = calc_cell_key(prnt_idx, prnt_lv)
            nhbr = nhbr_col.cells[nhbr_key]
            if nhbr.is_lf:
                nhbr_cells[0] = nhbr
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
                nhbr_cells[0] = nhbr_0
                
                nhbr_1 = nhbr_col.cells[nhbr_1_key]
            if nhbr_1.is_lf:
                nhbr_cells[1] = nhbr_1

    return nhbr_cells
