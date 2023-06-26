from .Error_Indicator_Column import Error_Indicator_Column
from .Error_Indicator_Cell import Error_Indicator_Cell

class Error_Indicator():
    def __init__(self, mesh, ref_form = None, ref_kind = None,
                 tol_spt = 1.1,   tol_ang = 1.1,
                 by_col  = False, by_cell = False):
        
        self.ref_form = ref_form
        self.ref_kind = ref_kind
        self.tol_spt  = tol_spt
        self.tol_ang  = tol_ang
        self.by_col   = by_col
        self.by_cell  = by_cell

        self.max_err  = 0.
        
        self.cols = {} # Columns in mesh
        col_items = sorted(mesh.cols.items())
        for col_key, col in col_items:
            if col.is_lf:
                self.cols[col_key] = Error_Indicator_Column(ref_form, ref_kind,
                                                            tol_spt, tol_ang,
                                                            by_col, by_cell)
                if by_cell:
                    cell_items = sorted(col.cells.items())
                    for cell_key, cell in cell_items:
                        if cell.is_lf:
                            self.cols[col_key].cells[cell_key] = Error_Indicator_Cell(
                                ref_form, ref_kind,
                                tol_spt, tol_ang,
                                by_col, by_cell)
        
    def __str__(self):
        msg = ( 'Hello world.'
               )
        
        return msg

class Error_Indicator_old():
    def __init__(self, mesh, by_col, by_cell):
        self.by_col  = by_col
        self.by_cell = by_cell
        self.max_err = 0.
        
        self.cols = {} # Columns in mesh
        col_items = sorted(mesh.cols.items())
        for col_key, col in col_items:
            if col.is_lf:
                self.cols[col_key] = Error_Indicator_Column(by_col, by_cell)
                if by_cell:
                    cell_items = sorted(col.cells.items())
                    for cell_key, cell in cell_items:
                        if cell.is_lf:
                            self.cols[col_key].cells[cell_key] = Error_Indicator_Cell(by_col,
                                                                                      by_cell)
        
    def __str__(self):
        msg = ( 'Hello world.'
               )
        
        return msg
