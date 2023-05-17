import copy
import sys

from .Column import Column

from .calc_key import calc_col_key

# Refine a column spatially

def ref_col_spt(self, col_key, form = 'h'):
    if (form == 'p') or (form == 'hp'):
        ref_col_spt_p(self, col_key)
        
    if (form == 'h') or (form == 'hp'):
        ref_col_spt_h(self, col_key)
        
    if form not in ['h', 'p', 'hp']:
        msg = ( 'ERROR IN REFINING COLUMN, ' +
                    'UNSUPPORTED REFINEMENT FORM - {}').format(form)
        print(msg)
        sys.exit(0)
            
def ref_col_spt_p(self, col_key):
    col = self.cols[col_key]
    
    if col.is_lf:
        col.ndofs[0] += 1
        col.ndofs[1] += 1
        
        # Check if neighbors need to be refined.
        for F in range(0, 4):
            unique_nhbr_keys = list(set(col.nhbr_keys[F]))
            for nhbr_key in unique_nhbr_keys:
                if nhbr_key is not None:
                    nhbr = self.cols[nhbr_key]
                    if nhbr.is_lf:
                        # If nhbr has matching neighbor flags, then it shares
                        # a lower-level neighbor and must be refined as well.
                        for Fp in range(0, 4):
                            if col.nhbr_keys[Fp] != [None, None]:
                                if col.nhbr_keys[Fp] == nhbr.nhbr_keys[Fp]:
                                    if ((nhbr.ndofs[0] != col.ndofs[0]) and
                                        (nhbr.ndofs[1] != col.ndofs[1])):
                                        ref_col_spt_p(self, nhbr_key)

    


def ref_col_spt_h(self, col_key):
    col = self.cols[col_key]
    
    if col.is_lf:
        [i, j]           = col.idx[:]
        lv               = col.lv
        [x0, y0, xf, yf] = col.pos[:]
        
        # Check if neighbors need to be refined first.
        for F in range(0, 4):
            unique_nhbr_keys = list(set(col.nhbr_keys[F]))
            for nhbr_key in unique_nhbr_keys:
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
        chldn_idxs = [[2 * i    , 2 * j    ],
                      [2 * i + 1, 2 * j    ],
                      [2 * i    , 2 * j + 1],
                      [2 * i + 1, 2 * j + 1]]
        
        x_mid = (x0 + xf) / 2.
        y_mid = (y0 + yf) / 2.
        chldn_poss = [
            [x0,    y0,    x_mid, y_mid],
            [x0,    y_mid, x_mid, yf   ],
            [x_mid, y0,    xf,    y_mid],
            [x_mid, y_mid, xf,    yf   ]
        ]
        
        chldn_keys = [0] * 4
        for ii in range(0, 4):
            chld_idx       = chldn_idxs[ii]
            chldn_keys[ii] = calc_col_key(chld_idx, lv + 1)
            
        # Neighbor keys of child 0 (bottom-left)
        chldn_nhbr_keys_0 = [[None, None], [None, None], [None, None], [None, None]]
        chldn_nhbr_keys_0[0] =     [chldn_keys[2],       chldn_keys[2]      ] # F = 0
        chldn_nhbr_keys_0[1] =     [chldn_keys[1],       chldn_keys[1]      ] # F = 1
        if col.nhbr_keys[2][1] != col_key:
            chldn_nhbr_keys_0[2] = [col.nhbr_keys[2][1], col.nhbr_keys[2][1]] # F = 2
        else: # Edge case, column is its own neighbor
            chldn_nhbr_keys_0[2] = [chldn_keys[2],       chldn_keys[2]      ] # F = 2
        if col.nhbr_keys[3][0] != col_key:
            chldn_nhbr_keys_0[3] = [col.nhbr_keys[3][0], col.nhbr_keys[3][0]] # F = 3
        else: # Edge case, column is its own neighbor
            chldn_nhbr_keys_0[3] = [chldn_keys[1],       chldn_keys[1]      ] # F = 3
        
        # Neighbor keys of child 1 (top-left)
        chldn_nhbr_keys_1 = [[None, None], [None, None], [None, None], [None, None]]
        chldn_nhbr_keys_1[0] =     [chldn_keys[3],       chldn_keys[3]      ] # F = 0
        if col.nhbr_keys[1][1] != col_key:
            chldn_nhbr_keys_1[1] = [col.nhbr_keys[1][1], col.nhbr_keys[1][1]] # F = 1
        else: # Edge case, column is its own neighbor
            chldn_nhbr_keys_1[1] = [chldn_keys[0],       chldn_keys[0]      ] # F = 1
        if col.nhbr_keys[2][0] != col_key:
            chldn_nhbr_keys_1[2] = [col.nhbr_keys[2][0], col.nhbr_keys[2][0]] # F = 2
        else: # Edge case, column is its own neighbor
            chldn_nhbr_keys_1[2] = [chldn_keys[3],       chldn_keys[3]      ] # F = 2
        chldn_nhbr_keys_1[3] =     [chldn_keys[0],       chldn_keys[0]      ] # F = 3
        
        # Neighbor keys of child 2 (bottom-right)
        chldn_nhbr_keys_2 = [[None, None], [None, None], [None, None], [None, None]]
        if col.nhbr_keys[0][0] != col_key:
            chldn_nhbr_keys_2[0] = [col.nhbr_keys[0][0], col.nhbr_keys[0][0]] # F = 0
        else: # Edge case, column is its own neighbor
            chldn_nhbr_keys_2[0] = [chldn_keys[0],       chldn_keys[0]      ] # F = 0
        chldn_nhbr_keys_2[1] =     [chldn_keys[3],       chldn_keys[3]      ] # F = 1
        chldn_nhbr_keys_2[2] =     [chldn_keys[0],       chldn_keys[0]      ] # F = 2
        if col.nhbr_keys[3][1] != col_key:
            chldn_nhbr_keys_2[3] = [col.nhbr_keys[3][1], col.nhbr_keys[3][1]] # F = 3
        else: # Edge case, column is its own neighbor
            chldn_nhbr_keys_2[3] = [chldn_keys[3],       chldn_keys[3]      ] # F = 3
        
        # Neighbor keys of child 3 (top-right)
        chldn_nhbr_keys_3 = [[None, None], [None, None], [None, None], [None, None]]
        if col.nhbr_keys[0][1] != col_key:
            chldn_nhbr_keys_3[0] = [col.nhbr_keys[0][1], col.nhbr_keys[0][1]] # F = 0
        else: # Edge case, column is its own neighbor
            chldn_nhbr_keys_3[0] = [chldn_keys[1],       chldn_keys[1]      ] # F = 0
        if col.nhbr_keys[1][0] != col_key:
            chldn_nhbr_keys_3[1] = [col.nhbr_keys[1][0], col.nhbr_keys[1][0]] # F = 1
        else: # Edge case, column is its own neighbor
            chldn_nhbr_keys_3[1] = [chldn_keys[2],       chldn_keys[2]      ] # F = 1
        chldn_nhbr_keys_3[2] =     [chldn_keys[1],       chldn_keys[1]      ] # F = 2
        chldn_nhbr_keys_3[3] =     [chldn_keys[2],       chldn_keys[2]      ] # F = 3

        chldn_nhbr_keys = [chldn_nhbr_keys_0,
                           chldn_nhbr_keys_1,
                           chldn_nhbr_keys_2,
                           chldn_nhbr_keys_3]
        
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
                
        # Update nhbr_keys of neighboring columns
        new_nhbr_chld_idxs = [2, 3, 1, 0]
        for F in range(0, 4):
            for nhbr_num in range(0, 2):
                nhbr_key = col.nhbr_keys[F][nhbr_num]
                if (nhbr_key is not None):
                    nhbr = self.cols[nhbr_key]
                    if nhbr.is_lf:
                        idx = 2 * F + nhbr_num
                        ii = new_nhbr_chld_idxs[int((idx + 1) / 2) % 4]
                        nhbr_key = chldn_keys[ii]
                        nhbr_lv = nhbr.lv
                        if nhbr_lv - lv == 0:
                            nhbr.nhbr_keys[(F + 2)%4][(nhbr_num + 1)%2] = nhbr_key
                        elif nhbr_lv- lv == 1:
                            nhbr.nhbr_keys[(F + 2)%4][0] = nhbr_key
                            nhbr.nhbr_keys[(F + 2)%4][1] = nhbr_key
                                
        self.del_col(col_key)
