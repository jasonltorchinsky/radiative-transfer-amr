class Projection_Column:
    def __init__(self, col, proj_cells):
        self.pos   = col.pos[:]    # Spatial corners of columns
        self.idx   = col.idx[:]    # Spatial index for column
        self.lv    = col.lv        # Level of spatial refinement for column
        self.key   = col.key       # Unique key for column
        self.is_lf = col.is_lf     # Whether cell is a leaf or not
        self.ndofs = col.ndofs[:]  # Degrees of freedom in x-, y-
        self.cells = proj_cells    # Dict of cells in the column
        

    def __str__(self):
        msg = ( "    Column:  {}, {}\n".format(self.idx, self.lv) +
                "       key:  {}\n".format(self.key) +
                "       pos:  {}\n".format(self.pos) +
                "     ndofs:  {}\n".format(self.ndofs) +
                "     cells:  {}\n".format(list(self.cells.keys()))
               )

        return msg
