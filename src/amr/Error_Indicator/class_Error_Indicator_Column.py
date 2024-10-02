import os, sys
src_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, os.pardir))

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports

class Error_Indicator_Column:
    def __init__(self, **kwargs):

        default_kwargs = {"ref_col"      : False,
                          "col_ref_form" : None,
                          "col_ref_kind" : None,
                          "col_ref_tol"  : None,
                          "ref_cell"      : False,
                          "cell_ref_form" : None,
                          "cell_ref_kind" : None,
                          "cell_ref_tol"  : None}
        kwargs = {**default_kwargs, **kwargs}

        # Do we refine columns? If so, how?
        self.ref_col  = kwargs["ref_col"]
        if self.ref_col:
            self.ref_form = kwargs["col_ref_form"]
            self.ref_kind = kwargs["col_ref_kind"]
            self.ref_tol  = kwargs["col_ref_tol"]
            self.err      = 0.
        else: # If we"re not refining columns, ignore column refinement paremeters
            self.ref_form = None
            self.ref_kind = None
            self.ref_tol  = None
            self.err      = None
        
        # Do we refine cells? If so, how?
        self.ref_cell      = kwargs["ref_cell"]
        self.cell_ref_form = kwargs["cell_ref_form"]
        self.cell_ref_kind = kwargs["cell_ref_kind"]
        self.cell_ref_tol  = kwargs["cell_ref_tol"]
        
        if self.ref_cell:
            self.cells = {}
        else:
            self.cells = None

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
