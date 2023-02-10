import numpy as np
import matplotlib.pyplot as plt # Only for LaTeX strings
import sys
import os

from .calc_key import calc_col_key, calc_cell_key
from .get_nhbr import get_col_nhbr, get_cell_nhbr

class Mesh:
    ''' Collection of columns.'''

    def __init__(self, Ls, pbcs, ndofs = [2, 2, 2], has_th = False):
        # We assume that the parameters here are valid
        self.Ls   = Ls # Lengths of spatial domain
        self.pbcs = pbcs # Periodicity of spatial domain
        self.has_th = has_th # Include angular domain in mesh?

        # Create first cell
        if has_th:
            cell = Cell(pos = [0, 2 * np.pi],
                        idx = 0,
                        lv = 0,
                        is_lf = True,
                        ndofs = [ndofs[2]],
                        quad = None)
        else:
            cell = Cell(pos = [0, 0],
                        idx = 0,
                        lv = 0,
                        is_lf = True,
                        ndofs = [0],
                        quad = None)

        # Create first column, put cell into it
        # Determine boundary flags for column
        col_bdry = [True, True, True, True] # Which face is on the boundary of
                                            # the domain?
        if pbcs[0]: # Periodic in x, no "boundary" in x
            col_bdry[0] = False
            col_bdry[2] = False

        if pbcs[1]: # Periodic in y, no "boundary" in y
            col_bdry[1] = False
            col_bdry[3] = False
            
        col = Column(pos = [0, 0, Ls[0], Ls[1]],
                     idx = [0, 0],
                     lv = 0,
                     is_lf = True,
                     ndofs = ndofs[0:2],
                     cells = {0 : cell},
                     bdry  = col_bdry)
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

    def ref_col(self, col):
        '''
        Refine a colmun, spatially.
        '''
        if col.is_lf:
            [i, j]  = col.idx[:]
            lv = col.lv
            [x0, y0, x1, y1] = col.pos

            # Add four columns that are repeats of current column
            chldn_idxs = [[2 * i    , 2 * j    ],
                          [2 * i + 1, 2 * j    ],
                          [2 * i    , 2 * j + 1],
                          [2 * i + 1, 2 * j + 1]]

            chldn_bdrys = [[False      , False      , col.bdry[2], col.bdry[3]],
                           [col.bdry[0], False      , False      , col.bdry[3]],
                           [False      , col.bdry[1], col.bdry[2], False      ],
                           [col.bdry[0], col.bdry[1], False      , False      ],
                           ]

            x_mid = (x0 + x1) / 2.
            y_mid = (y0 + y1) / 2.
            chldn_poss = [[x0,    y0,    x_mid, y_mid],
                          [x_mid, y0,    x1,    y_mid],
                          [x0,    y_mid, x_mid, y1  ],
                          [x_mid, y_mid, x1,    y1  ]]

            for ii in range(0, 4):
                chld_idx  = chldn_idxs[ii]
                chld_pos  = chldn_poss[ii]
                chld_bdry = chldn_bdrys[ii]
                chld_col = Column(pos = chld_pos,
                                  idx = chld_idx,
                                  lv  = lv + 1,
                                  is_lf = True,
                                  ndofs = col.ndofs,
                                  cells = col.cells.copy(),
                                  bdry  = chld_bdry)
                self.add_col(chld_col)

            # Make sure we maintain 2-neighbor condition
            for nhbr_loc in ['+', '-']:
                for axis in range(0, 2):
                    [flag, nhbr, _] = get_col_nhbr(self, col,
                                                   axis = axis,
                                                   nhbr_loc = nhbr_loc)
                    if (flag == 'pm') or (flag == 'pp'):
                        self.ref_col(nhbr)

            self.del_col(col)

    def ref_mesh(self):
        '''
        Refine each column in the mesh, spatially.
        '''
        keys = list(self.cols.keys())
        for key in keys:
            self.ref_col(self.cols[key])
        

class Column:
    ''' Column of cells. '''

    def __init__(self, pos, idx, lv, is_lf, ndofs, cells, bdry):
        self.pos   = pos   # Spatial corners of columns
        self.idx   = idx   # Spatial index for column
        self.lv    = lv    # Level of spatial refinement for column
        self.key   = calc_col_key(idx, lv) # Unique key for column
        self.is_lf = is_lf # Whether column is a leaf or not
        self.ndofs = ndofs # Degrees of freedom in x-, y-.
        self.cells = cells # List of cells in the column
        self.bdry  = bdry  # List of faces that lie on the domain boundary
        

    def __str__(self):
        msg = ( 'Column:  {}, {}\n'.format(self.idx, self.lv) +
                '   key:  {}\n'.format(self.key) +
                '   pos:  {}\n'.format(self.pos) +
                ' is_lf:  {}\n'.format(self.is_lf) +
                ' cells:  {}\n'.format(list(self.cells.keys())) +
                '  bdry:  {}\n'.format(self.bdry)
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
                    

            for ii in range(0, 2):
                chld_idx  = chldn_idxs[ii]
                chld_pos  = chldn_poss[ii]
                chld_quad = chldn_quads[ii]
                chld_cell = Cell(pos = chld_pos,
                                 idx = chld_idx,
                                 lv  = lv + 1,
                                 is_lf = True,
                                 ndofs = cell.ndofs[:],
                                 quad = chld_quad)
                self.add_cell(chld_cell)

            for nhbr_loc in ['+', '-']:
                nhbr = get_cell_nhbr(self, cell, nhbr_loc = nhbr_loc)
                if lv - nhbr.lv == 1:
                    self.ref_cell(nhbr)

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

    def __init__(self, pos, idx, lv, is_lf, ndofs, quad):
        self.pos   = pos    # Position in angular dimension
        self.idx   = idx    # Angular index of cell
        self.lv    = lv     # Level of angular refinement
        self.key   = calc_cell_key(idx, lv) # Unique key for cell
        self.is_lf = is_lf  # Whether cell is a leaf or not
        self.ndofs = ndofs  # Degrees of freedom in theta-.
        self.quad  = quad   # Which angular quadrant the cell is in
        
    def __str__(self):
        pos_str = ( '[{:3.2f} pi'.format(self.pos[0] / np.pi) +
                    ', {:3.2f} pi]'.format(self.pos[1] / np.pi)
                   )
        msg = ( 'Cell  :  {}, {}\n'.format(self.idx, self.lv) +
                ' key  :  {}\n'.format(self.key) +
                ' pos  :  {}\n'.format(pos_str) +
                ' is_lf:  {}\n'.format(self.is_lf) +
                ' ndofs:  {}\n'.format(self.ndofs) +
                '  quad:  {}\n'.format(self.quad)
               )

        return msg
