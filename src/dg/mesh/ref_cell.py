# Standard Library Imports
import sys

# Third-Party Library Imports

# Local Library Imports
import consts

# Relative Imports
from .column import Column

from .column.cell import Cell
from .column.cell import calc_key as calc_cell_key

def ref_cell(self, col_key: int, cell_key: int, form: str = "h") -> None:
    if form in ["p", "hp"]:
        ref_cell_p(self, col_key, cell_key)
    if form in ["h", "hp"]:
        ref_cell_h(self, col_key, cell_key)
    if form not in ["h", "p", "hp"]:
        msg: str = ( "ERROR IN REFINING CELL, " +
                     "UNSUPPORTED REFINEMENT FORM - {}").format(form)
        print(msg)
        sys.exit(0)

def ref_cell_p(self, col_key: int, cell_key: int) -> None:
    col: Column = self.cols[col_key]
    cell: Cell  = col.cells[cell_key]

    assert(cell.is_lf)
    cell.ndofs[0] += 1

    # Check if spatially neighboring cells need to be refined
    for col_face in range(0, 4):
        unique_nhbr_col_keys: list = list(set(col.nhbr_keys[col_face]))
        for nhbr_col_key in unique_nhbr_col_keys:
            if nhbr_col_key is not None:
                nhbr_col: Column = self.cols[nhbr_col_key]
                assert(nhbr_col.is_lf)
                
                nhbr_cell_keys: list = self.nhbr_cells_in_nhbr_col(col_key,
                                                                   cell_key,
                                                                   nhbr_col_key)
                
                unique_nhbr_cell_keys: list = list(set(nhbr_cell_keys))
                for nhbr_cell_key in unique_nhbr_cell_keys:
                    if nhbr_cell_key is not None:
                        nhbr_cell: Cell = nhbr_col.cells[nhbr_cell_key]
                        assert(nhbr_cell.is_lf)

                        if abs(cell.ndofs[0] - nhbr_cell.ndofs[0]) > 1:
                            ref_cell_p(self, nhbr_col_key, nhbr_cell_key)

    # Check if angularly neighboring cells need to be refined
    for nhbr_loc in ["+", "-"]:
        nhbr_cell: Cell = col.nhbr_cell(cell.key, nhbr_loc)
        if abs(cell.ndofs[0] - nhbr_cell.ndofs[0]) > 1:
            ref_cell_p(self, col_key, nhbr_cell.key)

def ref_cell_h(self, col_key: int, cell_key: int) -> None:
    col: Column = self.cols[col_key]
    cell: Cell  = col.cells[cell_key]
    
    assert(cell.is_lf)
    idx: int = cell.idx
    lv: int  = cell.lv
    [z0, zf] = cell.pos
    quadrant: int = cell.quadrant
        
    # Check if angularly neighboring cells need to be refined
    for F in range(0, 2):
        nhbr_cell_key: int = cell.nhbr_keys[F]
        if nhbr_cell_key is not None:
            nhbr_cell: Cell = col.cells[nhbr_cell_key]
            assert(nhbr_cell.is_lf)

            nhbr_cell_lv: int = nhbr_cell.lv
            if lv - nhbr_cell_lv == 1:
                self.ref_cell(col_key, nhbr_cell_key)
                        
    # Check if spatially neighboring cells need to be refined
    for F in range(0, 4):
        for nhbr_col_key in col.nhbr_keys[F]:
            if nhbr_col_key is not None:
                nhbr_col: Column = self.cols[nhbr_col_key]
                assert(nhbr_col.is_lf)
                
                nhbr_cell_keys: list = \
                    self.nhbr_cells_in_nhbr_col(col_key, cell_key,
                                                nhbr_col_key)
                for nhbr_cell_key in nhbr_cell_keys:
                    if nhbr_cell_key is not None:
                        nhbr_cell: Cell = nhbr_col.cells[nhbr_cell_key]
                        assert(nhbr_cell.is_lf)
                        
                        nhbr_cell_lv: int = nhbr_cell.lv
                        if lv - nhbr_cell_lv == 1:
                            self.ref_cell(nhbr_col_key, nhbr_cell_key)
        
    # Add two cells that are repeats of current cell
    chldn_idxs: list = [2 * idx    ,
                        2 * idx + 1]
        
    z_mid: float = (z0 + zf) / 2.
    chldn_poss: list = [[z0,    z_mid],
                        [z_mid, zf   ]]
        
    chldn_quadrants: list = [None, None]
    if quadrant != None:
        # If the parent cell is in an angular quadrant,
        # the child cells both will be
        chldn_quadrants: list = [quadrant, quadrant]
    else:
        # Check if each cell is in a quadrant
        S_quads: list = [[0, consts.PI/2], [consts.PI/2, consts.PI],
                         [consts.PI, 3*consts.PI/2], [3*consts.PI/2, 2*consts.PI]]
        for ii, chld_pos in enumerate(chldn_poss):
            for SS, S_quad in enumerate(S_quads):
                if (chld_pos[0] >= S_quad[0]) and (chld_pos[1] <= S_quad[1]):
                    chldn_quadrants[ii] = SS
                        
    chldn_keys: list = [0] * 2
    for ii in range(0, 2):
        chld_idx: int   = chldn_idxs[ii]
        chldn_keys[ii]  = calc_cell_key(chld_idx, lv + 1)
        
    chldn_nhbr_keys: list = [[None         , chldn_keys[1]],
                             [chldn_keys[0], None         ]]
        
    for F in range(0, 2):
        nhbr_key: int = cell.nhbr_keys[F]
        if nhbr_key is not None:
            nhbr: Cell = col.cells[nhbr_key]
            if nhbr_key == cell.key: # cell is own neighbor, special case
                chldn_nhbr_keys[F][F] = chldn_nhbr_keys[F][(F+1)%2]
            elif nhbr.is_lf:
                chldn_nhbr_keys[F][F] = cell.nhbr_keys[F]
            else:
                msg: str = ("ERROR IN MAKING CHILD CELLS, " +
                            "2-NEIGHBOR ASSUMPTION VIOLATED")
                print(msg)
                sys.exit(0)
        
    for ii in range(0, 2):
        chld_idx: int      = chldn_idxs[ii]
        chld_pos: list     = chldn_poss[ii]
        chld_quadrant: int  = chldn_quadrants[ii]
        chld_nhbr_keys: int = chldn_nhbr_keys[ii]
        chld_cell: Cell = Cell(pos   = chld_pos,
                               idx   = chld_idx,
                               lv    = lv + 1,
                               is_lf = True,
                               ndofs = cell.ndofs[:],
                               quadrant  = chld_quadrant,
                               nhbr_keys = chld_nhbr_keys[:])
        col.add_cell(chld_cell)
        
    # Also need to go to neighbor and update its keys
    for F, nhbr_key in enumerate(cell.nhbr_keys):
        if nhbr_key is not None:
            if nhbr_key != cell.key: # Make sure cell isn't self
                nhbr: Cell = col.cells[nhbr_key]
                assert(nhbr.is_lf)
                
                nhbr.nhbr_keys[(F+1)%2] = chldn_keys[F]
    
    col.del_cell(cell_key)
