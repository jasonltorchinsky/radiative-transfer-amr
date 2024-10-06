# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports

# Relative Imports

class Error_Indicator_Cell:
    def __init__(self, error: float, ref_form: str = "", do_ref: bool = False):
        self.error: float = error
        self.ref_form: str = ref_form
        self.do_ref: bool = do_ref
            
    def __str__(self):
        ref_cell_str  = (self.ref_cell  if self.ref_cell is not None else "None")
        ref_form_str  = (self.ref_form  if self.ref_form is not None else "None")
        ref_kind_str  = (self.ref_kind  if self.ref_kind is not None else "None")
        ref_tol_str   = (self.ref_tol   if self.ref_tol  is not None else "None")
        err_str       = ("{:.4E}".format(self.err) if self.err is not None else "None")
        
        msg = ( "Cell Refinement Information:\n" +
                "  ref_cell: {}\n".format(ref_cell_str) + 
                "  ref_form: {}\n".format(ref_form_str) +
                "  ref_kind: {}\n".format(ref_kind_str) +
                "  ref_tol:  {}\n".format(ref_tol_str) +
                "  err:      {}\n".format(err_str)
               )
        
        return msg
