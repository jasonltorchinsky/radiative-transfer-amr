import numpy as np
import os, sys

from .calc_key import calc_col_key, calc_cell_key
from .get_nhbr import get_cell_nhbr_in_col

class Mesh:
    ''' Collection of columns.'''

    def __init__(self, Ls, pbcs, ndofs = [2, 2, 2], has_th = False):
        # We assume that the parameters here are valid
        self.Ls     = Ls     # Lengths of spatial domain
        self.pbcs   = pbcs   # Periodicity of spatial domain
        self.has_th = has_th # Include angular domain in mesh?

        # Create first cell
        if has_th:
            cell = Cell(pos   = [0, 2 * np.pi],
                        idx   = 0,
                        lv    = 0,
                        is_lf = True,
                        ndofs = [ndofs[2]],
                        quad  = None,
                        nhbr_keys = [0, 0])
        else:
            cell = Cell(pos   = [0, 0],
                        idx   = 0,
                        lv    = 0,
                        is_lf = True,
                        ndofs = [0],
                        quad  = None,
                        nhbr_keys = None)

        # Create first column, put cell into it
        # Determine neighbor keys flags for column
        # F  => Right face, proceed counterclockwise
        nhbr_keys = [[None, None], [None, None], [None, None], [None, None]]
        # Which faces have a neighbor?
        if pbcs[0]: # Periodic in x, is own neighbor in x.
            nhbr_keys[0] = [0, None]
            nhbr_keys[2] = [0, None]

        if pbcs[1]: # Periodic in y, is own neighbor in y.
            nhbr_keys[1] = [0, None]
            nhbr_keys[3] = [0, None]
            
        col = Column(pos       = [0, 0, Ls[0], Ls[1]],
                     idx       = [0, 0],
                     lv        = 0,
                     is_lf     = True,
                     ndofs     = ndofs[0:2],
                     cells     = {0 : cell},
                     nhbr_keys = nhbr_keys)
        self.cols = {0 :  col} # Columns in mesh

    def __str__(self):
        msg = ( 'Ls     :  {}\n'.format(self.Ls) +
                'pbcs   :  {}\n'.format(self.pbcs) +
                'has_th :  {}\n'.format(self.has_th) +
                'cols   :  {}\n'.format(list(self.cols.keys()))
               )

        return msg

    def add_col(self, col):
        '''
        Add a column.
        '''
        self.cols[col.key] = col

    def del_col(self, col):
        '''
        Delete a column.
        '''
        del self.cols[col.key]

    def ref_col(self, col, kind = 'spt'):
        '''
        Refine a column, either spatially ('spt') or angularly ('ang').
        '''
        if col.is_lf:
            if kind == 'spt':
                [i, j]           = col.idx[:]
                lv               = col.lv
                [x0, y0, x1, y1] = col.pos

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
                                    self.ref_col(nhbr)
                                    
                                    
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
                chldn_nhbr_keys = [[[None, None         ], [chldn_keys[1], None], [chldn_keys[3], None], [None, None         ]],
                                   [[None, None         ], [None, None         ], [chldn_keys[2], None], [chldn_keys[0], None]],
                                   [[chldn_keys[1], None], [None, None         ], [None, None         ], [chldn_keys[3], None]],
                                   [[chldn_keys[0], None], [chldn_keys[2], None], [None, None         ], [None, None         ]]
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
                            print('ERROR IN MAKING CHILD COLUMNS, 2-NEIGHBOR ASSUMPTION VIOLATED')
                            sys.exit(0)
                                
                x_mid = (x0 + x1) / 2.
                y_mid = (y0 + y1) / 2.
                chldn_poss = [[x_mid, y0   , x1   , y_mid],
                              [x_mid, y_mid, x1   , y1   ],
                              [x0   , y_mid, x_mid, y1   ],
                              [x0   , y0   , x_mid, y_mid]
                              ]
                
                for ii in range(0, 4):
                    chld_idx       = chldn_idxs[ii]
                    chld_pos       = chldn_poss[ii]
                    chld_nhbr_keys = chldn_nhbr_keys[ii]
                    chld_col       = Column(pos = chld_pos,
                                            idx = chld_idx,
                                            lv  = lv + 1,
                                            is_lf = True,
                                            ndofs = col.ndofs,
                                            cells = col.cells.copy(),
                                            nhbr_keys  = chld_nhbr_keys)
                    self.add_col(chld_col)
                    
                # Also need to go to neighbor and update its keys
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
                                        print('ERROR IN REASSIGNING NEIGHBOR KEYS, 2-NEIGHBOR ASSUMPTION VIOLATED')
                                        sys.exit(0)
                                        
                self.del_col(col)

            elif kind == 'ang':
                keys = list(col.cells.keys())
                for key in keys:
                    self.ref_cell(col, col.cells[key])

            else:
                print('ERROR IN REFINING COLUMN, UNSUPPORTED REFINEMENT TYPE - {}'.format(kind))
                sys.exit(0)

    def ref_cell(self, col, cell):
        '''
        Refine a cell, angularly.
        '''
        if cell.is_lf:
            idx = cell.idx
            lv = cell.lv
            [z0, z1] = cell.pos
            quad = cell.quad

            # Check if angularly neighboring cells need to be refined
            for F in range(0, 2):
                for nhbr_cell_key in cell.nhbr_keys:
                    if nhbr_cell_key is not None:
                        nhbr_cell = col.cells[nhbr_cell_key]
                        if nhbr_cell.is_lf:
                            nhbr_cell_lv = nhbr_cell.lv
                            if lv - nhbr_cell_lv == 1:
                                self.ref_cell(col, nhbr_cell)

            # Check if spatially neighboring cells need to be refined
            for F in range(0, 4):
                for nhbr_col_key in col.nhbr_keys[F]:
                    if nhbr_col_key is not None:
                        nhbr_col = self.cols[nhbr_col_key]
                        if nhbr_col.is_lf:
                            nhbr_cells = get_cell_nhbr_in_col(cell, nhbr_col)
                            for nhbr_cell in nhbr_cells:
                                if nhbr_cell is not None:
                                    if nhbr_cell.is_lf:
                                        nhbr_cell_lv = nhbr_cell.lv
                                        if lv - nhbr_cell_lv == 1:
                                            self.ref_cell(nhbr_col, nhbr_cell)
                            
                            

            # Add two cells that are repeats of current cell
            chldn_idxs = [2 * idx    ,
                          2 * idx + 1]

            z_mid = (z0 + z1) / 2.
            chldn_poss = [[z0,    z_mid],
                          [z_mid, z1   ]]

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
                        print('ERROR IN MAKING CHILD CELLS, 2-NEIGHBOR ASSUMPTION VIOLATED')
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
                                      nhbr_keys = chld_nhbr_keys)
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
                                                
            col.del_cell(cell)

    def ref_mesh(self, kind = 'all'):
        '''
        Refine each column in the mesh, either spatially (spt), angularly (ang),
        or both ('all').
        '''

        keys = sorted(self.cols.keys())
        if (kind == 'ang') or (kind == 'all'):
            for key in keys:
                self.ref_col(self.cols[key], kind = 'ang')

        if (kind == 'spt') or (kind == 'all'):
            for key in keys:
                self.ref_col(self.cols[key], kind = 'spt')
        

class Column:
    ''' Column of cells. '''

    def __init__(self, pos, idx, lv, is_lf, ndofs, cells, nhbr_keys):
        self.pos   = pos   # Spatial corners of columns
        self.idx   = idx   # Spatial index for column
        self.lv    = lv    # Level of spatial refinement for column
        self.key   = calc_col_key(idx, lv) # Unique key for column
        self.is_lf = is_lf # Whether column is a leaf or not
        self.ndofs = ndofs # Degrees of freedom in x-, y-.
        self.cells = cells # Dict of cells in the column
        self.nhbr_keys = nhbr_keys  # List of neighbor keys
        

    def __str__(self):
        msg = ( 'Column    :  {}, {}\n'.format(self.idx, self.lv) +
                '       key:  {}\n'.format(self.key) +
                '       pos:  {}\n'.format(self.pos) +
                '     is_lf:  {}\n'.format(self.is_lf) +
                '     cells:  {}\n'.format(list(self.cells.keys())) +
                ' nhbr_keys:  {}\n'.format(self.nhbr_keys)
               )

        return msg

    def add_cell(self, cell):
        '''
        Add cell.
        '''
        self.cells[cell.key] = cell

    def del_cell(self, cell):
        '''
        Delete cell.
        '''
        del self.cells[cell.key]

    def ref_cell(self, cell):
        '''
        Refine a cell, angularly.
        '''
        if cell.is_lf:
            idx = cell.idx
            lv = cell.lv
            [z0, z1] = cell.pos
            quad = cell.quad

            # Check if angularly neighboring cells need to be refined
            for F in range (0, 2):
                for nhbr_key in cell.nhbr_keys:
                    if nhbr_key is not None:
                        nhbr = self.cells[nhbr_key]
                        if nhbr.is_lf:
                            nhbr_lv = nhbr.lv
                            if lv - nhbr_lv == 1:
                                self.ref_cell(nhbr)
                            

            # Add two columns that are repeats of current column
            chldn_idxs = [2 * idx    ,
                          2 * idx + 1]

            z_mid = (z0 + z1) / 2.
            chldn_poss = [[z0,    z_mid],
                          [z_mid, z1   ]]

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
                    nhbr = self.cells[nhbr_key]
                    if nhbr_key == cell.key: # cell is own neighbor, special case
                        chldn_nhbr_keys[F][F] = chldn_nhbr_keys[F][(F+1)%2]
                    elif nhbr.is_lf:
                        chldn_nhbr_keys[F][F] = cell.nhbr_keys[F]
                    else:
                        print('ERROR IN MAKING CHILD CELLS, 2-NEIGHBOR ASSUMPTION VIOLATED')
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
                                      nhbr_keys = chld_nhbr_keys)
                self.add_cell(chld_cell)

            # Also need to go to neighbor and update its keys
            for F, nhbr_key in enumerate(cell.nhbr_keys):
                if nhbr_key is not None:
                    if nhbr_key != cell.key: # Make sure cell isn't self
                        nhbr = self.cells[nhbr_key]
                        if nhbr.is_lf:
                            nhbr.nhbr_keys[(F+1)%2] = chldn_keys[F]

            # Make sure we maintain 2-neighbor condition
            # TO BE FIXED
                                                
            self.del_cell(cell)

    def ref_col(self):
        '''
        Refine all cells in a column, angularly.
        '''
        keys = list(self.cells.keys())
        for key in keys:
            self.ref_cell(self.cells[key])

    
class Cell:
    ''' Each individual cell. '''

    def __init__(self, pos, idx, lv, is_lf, ndofs, quad, nhbr_keys):
        self.pos   = pos    # Position in angular dimension
        self.idx   = idx    # Angular index of cell
        self.lv    = lv     # Level of angular refinement
        self.key   = calc_cell_key(idx, lv) # Unique key for cell
        self.is_lf = is_lf  # Whether cell is a leaf or not
        self.ndofs = ndofs  # Degrees of freedom in theta-.
        self.quad  = quad   # Which angular quadrant the cell is in.
        self.nhbr_keys = nhbr_keys # Keys for neighboring cells in column.
        
    def __str__(self):
        pos_str = ( '[{:3.2f} pi'.format(self.pos[0] / np.pi) +
                    ', {:3.2f} pi]'.format(self.pos[1] / np.pi)
                   )
        msg = ( '   Cell  :  {}, {}\n'.format(self.idx, self.lv) +
                '    key  :  {}\n'.format(self.key) +
                '    pos  :  {}\n'.format(pos_str) +
                '    is_lf:  {}\n'.format(self.is_lf) +
                '    ndofs:  {}\n'.format(self.ndofs) +
                '     quad:  {}\n'.format(self.quad) +
                'nhbr_keys:  {}\n'.format(self.nhbr_keys)
               )

        return msg
