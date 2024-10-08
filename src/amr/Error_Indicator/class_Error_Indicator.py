# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports
import consts
from dg.projection import Projection

# Relative Imports


class Error_Indicator():
    def __init__(self, proj: Projection, ref_kind: str = "all", 
                 ref_form: str = "hp", ref_tol: list = [consts.INF, consts.INF]):
        self.proj: Projection = proj
        self.ref_kind: str = ref_kind
        self.ref_form: str = ref_form
        [self.ang_ref_tol, self.spt_ref_tol] = ref_tol # [ang_ref_tol, spt_ref_tol]

    from .cell_hp_steer import cell_hp_steer
    from .col_hp_steer import col_hp_steer

    from .error import error_analytic, error_cell_jump, error_col_jump
    from .ref_by_ind import ref_by_ind

    def __str__(self):
        cols_str         = (sorted(list(self.cols.keys())) if self.cols is not None else "None")
        
        ref_col_str      = (self.ref_col      if self.ref_col      is not None else "None")
        col_ref_form_str = (self.col_ref_form if self.col_ref_form is not None else "None")
        col_ref_kind_str = (self.col_ref_kind if self.col_ref_kind is not None else "None")
        col_ref_tol_str  = (self.col_ref_tol  if self.col_ref_tol  is not None else "None")
        
        col_max_err_str  = ("{:.4E}".format(self.col_max_err) if self.col_max_err is not None else "None")
        
        ref_cell_str      = (self.ref_cell      if self.ref_cell      is not None else "None")
        cell_ref_form_str = (self.cell_ref_form if self.cell_ref_form is not None else "None")
        cell_ref_kind_str = (self.cell_ref_kind if self.cell_ref_kind is not None else "None")
        cell_ref_tol_str  = (self.cell_ref_tol  if self.cell_ref_tol  is not None else "None")
        
        cell_max_err_str = ("{:.4E}".format(self.cell_max_err) if self.cell_max_err is not None else "None")
        
        msg = ( "Columns: {}\n".format(cols_str) +
                "Column Refinement Information:\n" +
                "  ref_col:      {}\n".format(ref_col_str) + 
                "  col_ref_form: {}\n".format(col_ref_form_str) +
                "  col_ref_kind: {}\n".format(col_ref_kind_str) +
                "  col_ref_tol:  {}\n".format(col_ref_tol_str) +
                "  col_max_err:  {}\n".format(col_max_err_str) +
                "Cell Refinement Information:\n" +
                "  ref_cell:      {}\n".format(ref_cell_str) + 
                "  cell_ref_form: {}\n".format(cell_ref_form_str) +
                "  cell_ref_kind: {}\n".format(cell_ref_kind_str) +
                "  cell_ref_tol:  {}\n".format(cell_ref_tol_str) +
                "  cell_max_err:  {}\n".format(cell_max_err_str)
               )
        
        return msg
