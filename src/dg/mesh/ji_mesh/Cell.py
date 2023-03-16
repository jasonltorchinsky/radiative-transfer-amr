import numpy as np

from .calc_key import calc_cell_key

class Cell:
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
