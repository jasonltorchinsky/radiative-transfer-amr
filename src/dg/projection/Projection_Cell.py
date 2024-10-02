import numpy as np

class Projection_Cell():
    def __init__(self, cell, vals):
        self.pos   = cell.pos[:]   # Position in angular dimension
        self.idx   = cell.idx      # Angular index of cell
        self.lv    = cell.lv       # Level of angular refinement
        self.key   = cell.key      # Unique key for cell
        self.is_lf = cell.is_lf    # Whether cell is a leaf or not
        self.ndofs = cell.ndofs[:] # Degrees of freedom in theta-..

        self.vals = vals
        
    def __str__(self):
        pos_str = ( "[{:3.2f} pi".format(self.pos[0] / np.pi) +
                    ", {:3.2f} pi]".format(self.pos[1] / np.pi)
                   )
        msg = ( "   Cell  :  {}, {}\n".format(self.idx, self.lv) +
                "    key  :  {}\n".format(self.key) +
                "    pos  :  {}\n".format(pos_str) +
                "    ndofs:  {}\n".format(self.ndofs)
               )

        return msg
