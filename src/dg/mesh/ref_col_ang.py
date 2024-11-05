# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports
from .column import Column

# Relative Imports

def ref_col_ang(self, col_key: int):
    col: Column = self.cols[col_key]
    
    assert(col.is_lf)
    
    cell_keys: list = sorted(col.cells.keys())
    for cell_key in cell_keys:
        self.ref_cell(col_key, cell_key)
