from .calc_key import calc_col_key

class Column:
    def __init__(self, pos, idx, lv, is_lf, ndofs, cells, nhbr_keys):
        self.pos   = pos   # Spatial corners of columns
        self.idx   = idx   # Spatial index for column
        self.lv    = lv    # Level of spatial refinement for column
        self.key   = calc_col_key(idx, lv) # Unique key for column
        self.is_lf = is_lf # Whether column is a leaf or not
        self.ndofs = ndofs # Degrees of freedom in x-, y-.
        self.cells = cells # Dict of cells in the column
        self.nhbr_keys = nhbr_keys  # List of neighbor keys

    def __eq__(self, other):
        if self.key != other.key:
            return False
        if self.ndofs != other.ndofs: # Assume only one entry!
            return False
        if self.is_lf != other.is_lf:
            return False
        self_cell_keys = self.cells.keys()
        other_cell_keys = other.cells.keys()
        if self_cell_keys != other_cell_keys:
            return False
        for cell_key in self_cell_keys:
            if self.cells[cell_key] != other.cells[cell_key]:
                return False
        return True

    def __str__(self):
        msg = ( '    Column:  {}, {}\n'.format(self.idx, self.lv) +
                '       key:  {}\n'.format(self.key) +
                '       pos:  {}\n'.format(self.pos) +
                '     is_lf:  {}\n'.format(self.is_lf) +
                '     ndofs:  {}\n'.format(self.ndofs) +
                '     cells:  {}\n'.format(list(self.cells.keys())) +
                ' nhbr_keys:  {}\n'.format(self.nhbr_keys)
               )

        return msg

    def add_cell(self, cell):
        self.cells[cell.key] = cell

    def del_cell(self, cell_key):
        del self.cells[cell_key]
