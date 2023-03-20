import numpy as np

class Projection_Cell():
    def __init__(self, cell, vals):
        self.pos   = cell.pos # Position in angular dimension
        self.idx   = cell.idx # Angular index of cell
        self.lv    = cell.lv  # Level of angular refinement
        self.key   = cell.key # Unique key for cell
        self.ndofs = ndofs    # Degrees of freedom in theta-..

        self.vals = vals
        
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
