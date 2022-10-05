import numpy as np
import sys
import os


from .calc_key import calc_col_key, calc_cell_key
from .get_nhbr import get_col_nhbr, get_cell_nhbr

class Mesh:
    ''' Collection of columns.'''

    def __init__(self, Ls, pbcs, is_flat = False):
        # We assume that the parameters here are valid
        self.Ls   = Ls # Lengths of spatial domain
        self.pbcs = pbcs # Periodicity of spatial domain
        self.is_flat = is_flat
        if is_flat:
            cell = Cell(pos = [0, 0],
                        idx = 0,
                        lv = 0,
                        is_lf = True,
                        ndofs = [2, 2, 0])
        else:
            cell = Cell(pos = [0, 2 * np.pi],
                        idx = 0,
                        lv = 0,
                        is_lf = True,
                        ndofs = [2, 2, 2])

        col = Column(pos = [0, 0, Ls[0], Ls[1]],
                     idx = [0, 0],
                     lv = 0,
                     is_lf = True,
                     cells = {0 : cell})
        self.cols = {0 :  col} # Columns in mesh

    def __str__(self):
        msg = ( 'Ls     :  {}\n'.format(self.Ls) +
                'pbcs   :  {}\n'.format(self.pbcs) +
                'is_flat:  {}\n'.format(self.is_flat) +
                'cols   :  {}\n'.format(list(self.cols.keys()))
               )

        return msg

    def add_col(self, col):
        self.cols[col.key] = col

    def del_col(self, col):
        del self.cols[col.key]

    def ref_col(self, col):
        if col.is_lf:
            [i, j]  = col.idx[:]
            lv = col.lv
            [x0, y0, x1, y1] = col.pos

            # Add four columns that are repeats of current column
            chldn_idxs = [[2 * i,     2 * j    ],
                          [2 * i + 1, 2 * j    ],
                          [2 * i,     2 * j + 1],
                          [2 * i + 1, 2 * j + 1]]

            x_mid = (x0 + x1) / 2.
            y_mid = (y0 + y1) / 2.
            chldn_poss = [[x0,    y0,    x_mid, y_mid],
                          [x_mid, y0,    x1,    y_mid],
                          [x0,    y_mid, x_mid, y1  ],
                          [x_mid, y_mid, x1,    y1  ]]

            for ii in range(0, 4):
                chld_idx = chldn_idxs[ii]
                chld_pos = chldn_poss[ii]
                chld_col = Column(pos = chld_pos,
                                  idx = chld_idx,
                                  lv  = lv + 1,
                                  is_lf = True,
                                  cells = col.cells.copy())
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
        keys = list(self.cols.keys())
        for key in keys:
            self.ref_col(self.cols[key])
        

class Column:
    ''' Column of cells. '''

    def __init__(self, pos, idx, lv, is_lf, cells):
        self.pos   = pos   # Spatial corners of columns
        self.idx   = idx   # Spatial index for column
        self.lv    = lv    # Level of spatial refinement for column
        self.key   = calc_col_key(idx, lv) # Unique key for column
        self.is_lf = is_lf # Whether column is a leaf or not
        self.cells = cells # List of cells in the column
        

    def __str__(self):
        msg = ( 'Column:  {}, {}\n'.format(self.idx, self.lv) +
                '   key:  {}\n'.format(self.key) +
                '   pos:  {}\n'.format(self.pos) +
                ' is_lf:  {}\n'.format(self.is_lf) +
                ' cells:  {}\n'.format(list(self.cells.keys()))
               )

        return msg

    def add_cell(self, cell):
        self.cells[cell.key] = cell

    def del_cell(self, cell):
        del self.cells[cell.key]

    def ref_cell(self, cell):
        if cell.is_lf:
            idx = cell.idx
            lv = cell.lv
            [z0, z1] = cell.pos

            # Add two columns that are repeats of current column
            chldn_idxs = [2 * idx    ,
                          2 * idx + 1]

            z_mid = (z0 + z1) / 2.
            chldn_poss = [[z0,    z_mid],
                          [z_mid, z1   ]]

            for ii in range(0, 2):
                chld_idx = chldn_idxs[ii]
                chld_pos = chldn_poss[ii]
                chld_cell = Cell(pos = chld_pos,
                                 idx = chld_idx,
                                 lv  = lv + 1,
                                 is_lf = True,
                                 ndofs = cell.ndofs[:])
                self.add_cell(chld_cell)

            for nhbr_loc in ['+', '-']:
                nhbr = get_cell_nhbr(self, cell, nhbr_loc = nhbr_loc)
                if lv - nhbr.lv == 1:
                    self.ref_cell(nhbr)

            self.del_cell(cell)

    def ref_col(self):
        keys = list(self.cells.keys())
        for key in keys:
            self.ref_cell(self.cells[key])

    
class Cell:
    ''' Each individual cell. '''

    def __init__(self, pos, idx, lv, is_lf, ndofs):
        self.pos   = pos    # Position in angular dimension
        self.idx   = idx    # Angular index of cell
        self.lv    = lv     # Level of angular refinement
        self.key   = calc_cell_key(idx, lv) # Unique key for cell
        self.is_lf = is_lf  # Whether cell is a leaf or not
        self.ndofs = ndofs  # Degrees of freedom in x-, y-, theta-.
        
    def __str__(self):
        msg = ( 'Cell  :  {}, {}\n'.format(self.idx, self.lv) +
                ' key  :  {}\n'.format(self.key) +
                ' pos  :  {}\n'.format(self.pos) +
                ' is_lf:  {}\n'.format(self.is_lf) +
                ' ndofs:  {}\n'.format(self.ndofs)
               )

        return msg
