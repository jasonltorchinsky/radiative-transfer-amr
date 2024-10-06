# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports

# Relative Imports

class Error_Indicator_Column:
    def __init__(self, error: float, cells: dict, ref_form: str = "", 
                 do_ref: bool = False):
        self.error: float = error
        self.cells: dict = cells

        self.ref_form: str = ref_form
        self.do_ref: bool = do_ref

    def __str__(self):
        ref_col_str  = (self.ref_col  if self.ref_col  is not None else "None")
        ref_form_str = (self.ref_form if self.ref_form is not None else "None")
        ref_kind_str = (self.ref_kind if self.ref_kind is not None else "None")
        ref_tol_str  = (self.ref_tol  if self.ref_tol  is not None else "None")
        err_str      = ("{:.4E}".format(self.err) if self.err is not None else "None")
        
        cells_str         = (sorted(list(self.cells.keys())) if self.cells is not None else "None")
        
        ref_cell_str      = (self.ref_cell      if self.ref_cell is not None else "None")
        cell_ref_form_str = (self.cell_ref_form if self.ref_form is not None else "None")
        cell_ref_kind_str = (self.cell_ref_kind if self.ref_kind is not None else "None")
        cell_ref_tol_str  = (self.cell_ref_tol  if self.ref_tol  is not None else "None")
        
        msg = ( "Column Refinement Information:\n" +
                "  ref_col:  {}\n".format(ref_col_str) + 
                "  ref_form: {}\n".format(ref_form_str) +
                "  ref_kind: {}\n".format(ref_kind_str) +
                "  ref_tol:  {}\n".format(ref_tol_str) +
                "  err:      {}\n".format(err_str) +
                "Cells: {}\n".format(cells_str) +
                "Cell Refinement Information:\n" +
                "  ref_cell:      {}\n".format(ref_cell_str) + 
                "  cell_ref_form: {}\n".format(cell_ref_form_str) +
                "  cell_ref_kind: {}\n".format(cell_ref_kind_str) +
                "  cell_ref_tol:  {}\n".format(cell_ref_tol_str)
               )

        return msg
