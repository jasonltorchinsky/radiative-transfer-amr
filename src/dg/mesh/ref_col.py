# Standard Library Imports
import sys

# Third-Party Library Imports

# Local Library Imports

# Relative Imports
from .column import Column

def ref_col(self, col_key: int, kind: str = "all", form: str = "h") -> None:
    col: Column = self.cols[col_key]
    assert(col.is_lf)
    
    if kind in ["ang", "all"]:
        cell_keys: list = sorted(col.cells.keys())
        for cell_key in cell_keys:
            self.ref_cell(col_key, cell_key, form = form)
            
    if kind in ["spt", "all"]:
        self.ref_col_spt(col_key, form = form)
        
    if kind not in ["ang", "spt", "all"]:
        msg: str = ( "ERROR IN REFINING COLUMN, " +
                     "UNSUPPORTED REFINEMENT KIND - {}").format(kind)
        print(msg)
        sys.exit(0)
