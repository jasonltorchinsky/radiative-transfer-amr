# Standard Library Imports
import sys

# Third-Party Library Imports

# Local Library Imports

# Relative Imports
from .column import Column

def ref_col(self, col_key: int, kind: str = "all", form: str = "h") -> None:
    col: Column = self.cols[col_key]
    if col.is_lf:     
        if (kind == "ang") or (kind == "all"):
            cell_keys = sorted(col.cells.keys())
            for cell_key in cell_keys:
                self.ref_cell(col_key, cell_key, form = form)
                
        if (kind == "spt") or (kind == "all"):
            self.ref_col_spt(col_key, form = form)
            
        if (kind != "spt") and (kind != "ang") and (kind != "all"):
            msg: str = ( "ERROR IN REFINING COLUMN, " +
                         "UNSUPPORTED REFINEMENT KIND - {}").format(kind)
            print(msg)
            sys.exit(0)
