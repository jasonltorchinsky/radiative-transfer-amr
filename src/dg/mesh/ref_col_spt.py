import copy
import sys

from .Column import Column

from .calc_key import calc_col_key

# Refine a column spatially
def ref_col_spt(self, col_key):
    col = self.cols[col_key]
    
    if col.is_lf:
        [i, j]           = col.idx[:]
        lv               = col.lv
        [x0, y0, xf, yf] = col.pos[:]
        
        # Check if neighbors need to be refined first.
        for F in range(0, 4):
            for nhbr_key in col.nhbr_keys[F]:
                if nhbr_key is not None:
                    nhbr = self.cols[nhbr_key]
                    if nhbr.is_lf:
                        nhbr_lv = nhbr.lv
                        
                        # If neighbor is one level more coarse,
                        # it must be refined first
                        if lv - nhbr_lv == 1:
                            self.ref_col(nhbr_key, kind = 'spt')
                            
                            
        # Add four columns that are repeats of current column with
        # different domain and neighbors
        chldn_idxs = [[2 * i    , 2 * j + 1],
                      [2 * i + 1, 2 * j + 1],
                      [2 * i + 1, 2 * j    ],
                      [2 * i    , 2 * j    ]]
        
        chldn_keys = [0] * 4
        for ii in range(0, 4):
            chld_idx       = chldn_idxs[ii]
            chldn_keys[ii] = calc_col_key(chld_idx, lv + 1)
            
        # The children neighbor keys depend on the levels of the neighbors.
        chldn_nhbr_keys = [
            [[None, None], [chldn_keys[1], None], [chldn_keys[3], None], [None, None]],
            [[None, None], [None, None], [chldn_keys[2], None], [chldn_keys[0], None]],
            [[chldn_keys[1], None], [None, None], [None, None], [chldn_keys[3], None]],
            [[chldn_keys[0], None], [chldn_keys[2], None], [None, None], [None, None]]
        ]
        
        for F in range(0, 4):
            # We only need to check the level of the first neighbor.
            # If there are two neighbors, they are the same level.
            nhbr_key = col.nhbr_keys[F][0]
            if nhbr_key is not None:
                nhbr = self.cols[nhbr_key]
                if nhbr_key == col.key: # col is own neighbor, special case
                    chldn_nhbr_keys[F][F]       = chldn_nhbr_keys[F][(F+2)%4][:]
                    chldn_nhbr_keys[(F+1)%4][F] = chldn_nhbr_keys[(F+1)%4][(F+2)%4][:]
                elif nhbr.is_lf:
                    nhbr_lv = nhbr.lv
                    if nhbr_lv == lv: # Refining current column,
                        # so single neighbor.
                        # Note: [a][b] => [child #][face of child]
                        chldn_nhbr_keys[F][F]       = col.nhbr_keys[F][:]
                        chldn_nhbr_keys[(F+1)%4][F] = col.nhbr_keys[F][:]
                    elif nhbr_lv - lv == 1: # Neighbor is one level more refined,
                        # so has two neighbors.
                        chldn_nhbr_keys[F][F]       = [col.nhbr_keys[F][0], None]
                        chldn_nhbr_keys[(F+1)%4][F] = [col.nhbr_keys[F][1], None]
                    else:
                        msg = ('ERROR IN MAKING CHILD COLUMNS, ' +
                               '2-NEIGHBOR ASSUMPTION VIOLATED')
                        print(msg)
                        sys.exit(0)
                            
        x_mid = (x0 + xf) / 2.
        y_mid = (y0 + yf) / 2.
        chldn_poss = [
            [x_mid, y0   , xf   , y_mid],
            [x_mid, y_mid, xf   , yf   ],
            [x0   , y_mid, x_mid, yf   ],
            [x0   , y0   , x_mid, y_mid]
        ]
        
        for ii in range(0, 4):
            chld_idx       = chldn_idxs[ii]
            chld_pos       = chldn_poss[ii]
            chld_nhbr_keys = chldn_nhbr_keys[ii]
            chld_col       = Column(pos   = chld_pos,
                                    idx   = chld_idx[:],
                                    lv    = lv + 1,
                                    is_lf = True,
                                    ndofs = col.ndofs[:],
                                    cells = copy.deepcopy(col.cells),
                                    nhbr_keys  = chld_nhbr_keys[:])
            self.add_col(chld_col)
                
        # Update keys of neighboring columns
        for F in range(0, 4):
            # We only need to check the level of the first neighbor.
            # If there are two neighbors, they are the same level.
            for nhbr_num, nhbr_key in enumerate(col.nhbr_keys[F]):
                if nhbr_key is not None:
                    if nhbr_key != col.key: # Make sure column isn't self.
                        nhbr = self.cols[nhbr_key]
                        if nhbr.is_lf:
                            nhbr_lv = nhbr.lv
                            if nhbr_lv == lv: # Refining current column,
                                # so single neighbor now has two neighbors
                                # Note: [a][b] => [child #][face of child]
                                # Order matters here, we go counterclockwise
                                # from the POV of the column of interest
                                key_0 = F
                                key_1 = (F + 1) % 4
                                
                                nhbr_keys = [chldn_keys[key_1], chldn_keys[key_0]]
                                
                                nhbr.nhbr_keys[(F + 2)%4] = nhbr_keys
                                
                            elif nhbr_lv - lv == 1: # Neighbor is one level more refined,
                                # so two neighbors now each have one.
                                key = (F + nhbr_num) % 4
                                nhbr.nhbr_keys[(F + 2)%4] = [chldn_keys[key], None]
                            else:
                                msg = ('ERROR IN REASSIGNING NEIGHBOR KEYS, ' +
                                       '2-NEIGHBOR ASSUMPTION VIOLATED')
                                print()
                                sys.exit(0)
                                
        self.del_col(col_key)
