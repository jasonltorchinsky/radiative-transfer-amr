import numpy as np
import sys

from .Cell import Cell

from .get_nhbr import get_cell_nhbr_in_col
from .calc_key import calc_cell_key

# Refine a cell angularly

def ref_cell(self, col_key, cell_key, form = 'h'):
    if form in ['p', 'hp']:
        ref_cell_p(self, col_key, cell_key)
    if form in ['h', 'hp']:
        ref_cell_h(self, col_key, cell_key)
    if form not in ['h', 'p', 'hp']:
        msg = ( 'ERROR IN REFINING CELL, ' +
                    'UNSUPPORTED REFINEMENT FORM - {}').format(form)
        print(msg)
        sys.exit(0)

def ref_cell_p(self, col_key, cell_key):
    col  = self.cols[col_key]
    cell = col.cells[cell_key]

    # The naming conventions break down here, leading to some atrocious code.
    if cell.is_lf:
        cell.ndofs[0] += 1
        # Check if angularly neighboring cells need to be refined
        '''
        for FC in range(0, 4):
            unique_nhbr_col_keys = list(set(col.nhbr_keys[FC]))
            for nhbr_col_key in unique_nhbr_col_keys:
                nhbr_col = self.cols[nhbr_col_key]
                if nhbr_col.is_lf:
                    nhbr_col_nhbr_cells = get_cell_nhbr_in_col(self,
                                                               col_key,
                                                               cell_key,
                                                               nhbr_col_key)
                    for Fc in range(0, 2):
                        nhbr_cell_key = cell.nhbr_keys[F]
                        if nhbr_cell_key is not None:
                            nhbr_cell = col.cells[nhbr_cell_key]
                            if nhbr_cell.is_lf:
                                nhbr_nhbr_col_nhbr_cells = \
                                    get_cell_nhbr_in_col(self,
                                                         col_key,
                                                         nhbr_cell_key,
                                                         nhbr_col_key)
                                if nhbr_col_nhbr_cells == nhbr_nhbr_col_nhbr_cells:
                                    if cell.ndofs[0] != nhbr_cell.ndofs[0]:
                                        ref_cell_p(self, col_key, nhbr_cell_key)
        '''

def ref_cell_h(self, col_key, cell_key):
    col  = self.cols[col_key]
    cell = col.cells[cell_key]
    
    if cell.is_lf:
        idx      = cell.idx
        lv       = cell.lv
        [z0, zf] = cell.pos
        quad     = cell.quad
        
        # Check if angularly neighboring cells need to be refined
        for F in range(0, 2):
            nhbr_cell_key = cell.nhbr_keys[F]
            if nhbr_cell_key is not None:
                nhbr_cell = col.cells[nhbr_cell_key]
                if nhbr_cell.is_lf:
                    nhbr_cell_lv = nhbr_cell.lv
                    if lv - nhbr_cell_lv == 1:
                        self.ref_cell(col_key, nhbr_cell_key)
                        
        # Check if spatially neighboring cells need to be refined
        for F in range(0, 4):
            for nhbr_col_key in col.nhbr_keys[F]:
                if nhbr_col_key is not None:
                    nhbr_col = self.cols[nhbr_col_key]
                    if nhbr_col.is_lf:
                        nhbr_cell_keys = get_cell_nhbr_in_col(self,
                                                          col_key,
                                                          cell_key,
                                                          nhbr_col_key)
                        for nhbr_cell_key in nhbr_cell_keys:
                            if nhbr_cell_key is not None:
                                nhbr_cell = nhbr_col.cells[nhbr_cell_key]
                                if nhbr_cell.is_lf:
                                    nhbr_cell_lv = nhbr_cell.lv
                                    if lv - nhbr_cell_lv == 1:
                                        self.ref_cell(nhbr_col_key, nhbr_cell_key)
        
        # Add two cells that are repeats of current cell
        chldn_idxs = [2 * idx    ,
                      2 * idx + 1]
        
        z_mid = (z0 + zf) / 2.
        chldn_poss = [[z0,    z_mid],
                      [z_mid, zf   ]]
        
        chldn_quads = [None, None]
        if quad != None:
            # If the parent cell is in an angular quadrant,
            # the child cells both will be
            chldn_quads = [quad, quad]
        else:
            # Check if each cell is in a quadrant
            S_quads = [[0, np.pi/2], [np.pi/2, np.pi],
                       [np.pi, 3*np.pi/2], [3*np.pi/2, 2*np.pi]]
            for ii, chld_pos in enumerate(chldn_poss):
                for SS, S_quad in enumerate(S_quads):
                    if (chld_pos[0] >= S_quad[0]) and (chld_pos[1] <= S_quad[1]):
                        chldn_quads[ii] = SS
                        
        chldn_keys = [0] * 2
        for ii in range(0, 2):
            chld_idx        = chldn_idxs[ii]
            chldn_keys[ii]  = calc_cell_key(chld_idx, lv + 1)
            
        chldn_nhbr_keys = [[None         , chldn_keys[1]],
                           [chldn_keys[0], None         ]
                           ]
        
        for F in range(0, 2):
            nhbr_key = cell.nhbr_keys[F]
            if nhbr_key is not None:
                nhbr = col.cells[nhbr_key]
                if nhbr_key == cell.key: # cell is own neighbor, special case
                    chldn_nhbr_keys[F][F] = chldn_nhbr_keys[F][(F+1)%2]
                elif nhbr.is_lf:
                    chldn_nhbr_keys[F][F] = cell.nhbr_keys[F]
                else:
                    msg = ('ERROR IN MAKING CHILD CELLS, ' +
                           '2-NEIGHBOR ASSUMPTION VIOLATED')
                    print(msg)
                    sys.exit(0)
        
        for ii in range(0, 2):
            chld_idx       = chldn_idxs[ii]
            chld_pos       = chldn_poss[ii]
            chld_quad      = chldn_quads[ii]
            chld_nhbr_keys = chldn_nhbr_keys[ii]
            chld_cell      = Cell(pos   = chld_pos,
                                  idx   = chld_idx,
                                  lv    = lv + 1,
                                  is_lf = True,
                                  ndofs = cell.ndofs[:],
                                  quad  = chld_quad,
                                  nhbr_keys = chld_nhbr_keys[:])
            col.add_cell(chld_cell)
        
        # Also need to go to neighbor and update its keys
        for F, nhbr_key in enumerate(cell.nhbr_keys):
            if nhbr_key is not None:
                if nhbr_key != cell.key: # Make sure cell isn't self
                    nhbr = col.cells[nhbr_key]
                    if nhbr.is_lf:
                        nhbr.nhbr_keys[(F+1)%2] = chldn_keys[F]
        
        # Make sure we maintain 2-neighbor condition
        # TO BE FIXED
        
        col.del_cell(cell_key)
