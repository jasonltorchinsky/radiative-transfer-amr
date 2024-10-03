# Standard Library Imports
from math import floor

# Third-Party Library Imports

# Local Library Imports
from class_Column import Column
from class_Column import calc_key as calc_col_key

# Relative Imports

def nhbr_cols(self, col_key: int, axis: int = 0, nhbr_loc: str = "+") -> list:
    # Get what the idx, lv of the neighbor would be
    col: Column = self.cols[col_key]
    idx: list = col.idx[:]
    lv: int   = col.lv
    
    nhbr_idx: list = idx[:]
    if nhbr_loc == "+":
        nhbr_idx[axis] += 1
    elif nhbr_loc == "-":
        nhbr_idx[axis] -= 1

    if self.pbcs[axis]:
        nhbr_idx[axis] = nhbr_idx[axis] % (2 ** lv)

    # If we look for a neighbor outside of the mesh, we should report no-neighbor
    if (nhbr_idx[axis] < 0) or (nhbr_idx[axis] >= (2 ** lv)):
        flag: str = "nn"
        nhbr_1: Column = None
        nhbr_2: Column = None
        
        return [flag, nhbr_1, nhbr_2]
    
    nhbr_key: int = calc_col_key(nhbr_idx, lv)
    # If neighbor is a leaf, we"re done
    try: # Instead of searching through keys of mesh.is_lf, we just go for it
        nhbr: Column = self.cols[nhbr_key]
        if nhbr.is_lf:
            flag: str = "f0"
            nhbr_1: Column = nhbr
            nhbr_2: Column = None
            
            return [flag, nhbr_1, nhbr_2]
    except:
        None
    
    # Neighbor is not leaf. Check if neighbor"s parent is leaf
    # Only need to check is cell is refined at least twice
    if lv >= 2:
        nhbr_prnt_idx: list = [floor(nhbr_idx[0] / 2), floor(nhbr_idx[1] / 2)]
        nhbr_prnt_key: int  = calc_col_key(nhbr_prnt_idx, lv - 1)
        # If neighbor"s parent is a leaf, we"re done
        try: # Instead of searching through keys of mesh.is_lf, we just go for it
            nhbr_prnt: Column = self.cols[nhbr_prnt_key]
            if nhbr_prnt.is_lf:
                if (nhbr_idx[axis] == 2 * floor(nhbr_idx[axis] / 2)):
                    flag: str = "pm"
                else:
                    flag: str = "pp"
                    
                nhbr_1: Column = nhbr_prnt
                nhbr_2: Column = None
            
                return [flag, nhbr_1, nhbr_2]
        except:
            None

    # Neighbor"s parent is not a leaf. Neighbor"s children must be leaves
    if axis == 0:
        if nhbr_loc == "+":
            nhbr_1_idx: list = [2 * nhbr_idx[0], 2 * nhbr_idx[1]]
            nhbr_2_idx: list = [2 * nhbr_idx[0], 2 * nhbr_idx[1] + 1]
        elif nhbr_loc == "-":
            nhbr_1_idx: list = [2 * nhbr_idx[0] + 1, 2 * nhbr_idx[1]]
            nhbr_2_idx: list = [2 * nhbr_idx[0] + 1, 2 * nhbr_idx[1] + 1]
    elif axis == 1:
        if nhbr_loc == "+":
            nhbr_1_idx: list = [2 * nhbr_idx[0],    2 * nhbr_idx[1]]
            nhbr_2_idx: list = [2 * nhbr_idx[0]+ 1, 2 * nhbr_idx[1]]
        elif nhbr_loc == "-":
            nhbr_1_idx: list = [2 * nhbr_idx[0],     2 * nhbr_idx[1] + 1]
            nhbr_2_idx: list = [2 * nhbr_idx[0] + 1, 2 * nhbr_idx[1] + 1]

    nhbr_1_key: int = calc_col_key(nhbr_1_idx, lv + 1)
    nhbr_2_key: int = calc_col_key(nhbr_2_idx, lv + 1)
    # If neighbor's children are leaves, we're done
    try: # Instead of searching through keys of mesh.is_lf, we just go for it
        nhbr_1: Column = self.cols[nhbr_1_key]
        nhbr_2: Column = self.cols[nhbr_2_key]
        if nhbr_1.is_lf and nhbr_2.is_lf:
            flag: str = "cc"
            nhbr_1: Column = nhbr_1
            nhbr_2: Column = nhbr_2
            
            return [flag, nhbr_1, nhbr_2]
    except:
        None
    
    # Otherwise, there's no neighbor
    flag: str = "nn"
    nhbr_1: Column = None
    nhbr_2: Column = None

    return [flag, nhbr_1, nhbr_2]