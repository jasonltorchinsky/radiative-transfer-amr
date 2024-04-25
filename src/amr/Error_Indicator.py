from .Error_Indicator_Column import Error_Indicator_Column
from .Error_Indicator_Cell import Error_Indicator_Cell

class Error_Indicator():
    def __init__(self, mesh, **kwargs):

        default_kwargs = {'ref_col'      : False,
                          'col_ref_form' : None,
                          'col_ref_kind' : None,
                          'col_ref_tol'  : None,
                          'ref_cell'      : False,
                          'cell_ref_form' : None,
                          'cell_ref_kind' : None,
                          'cell_ref_tol'  : None}
        kwargs = {**default_kwargs, **kwargs}

        # Do we refine columns? If so, how?
        self.ref_col      = kwargs['ref_col']
        self.col_ref_form = kwargs['col_ref_form']
        self.col_ref_kind = kwargs['col_ref_kind']
        self.col_ref_tol  = kwargs['col_ref_tol']
        
        if self.ref_col:
            self.col_max_err = 0.
            self.avg_col_ref_err = 0.
        else:
            self.col_max_err = None
            self.avg_col_ref_err = None
            
        # Do we refine cells? If so, how?
        self.ref_cell      = kwargs['ref_cell']
        self.cell_ref_form = kwargs['cell_ref_form']
        self.cell_ref_kind = kwargs['cell_ref_kind']
        self.cell_ref_tol  = kwargs['cell_ref_tol']
        
        if self.ref_cell:
            self.cell_max_err = 0.
            self.avg_cell_ref_err = 0.
        else:
            self.cell_max_err = None
            self.avg_cell_ref_err = None
        
        
        # Error indicator is structured similarly to mesh object
        self.cols = {}
        col_items = sorted(mesh.cols.items())
        for col_key, col in col_items:
            if col.is_lf:
                # Even if we don't refine by column, they're used to store cells
                self.cols[col_key] = Error_Indicator_Column(**kwargs)
                if self.ref_cell:
                    cell_items = sorted(col.cells.items())
                    for cell_key, cell in cell_items:
                        if cell.is_lf:
                            self.cols[col_key].cells[cell_key] \
                                = Error_Indicator_Cell(**kwargs)
        
    def __str__(self):
        cols_str        = (sorted(list(self.cols.keys())) if self.cols is not None else 'None')
        
        ref_col_str      = (self.ref_col      if self.ref_col      is not None else 'None')
        col_ref_form_str = (self.col_ref_form if self.col_ref_form is not None else 'None')
        col_ref_kind_str = (self.col_ref_kind if self.col_ref_kind is not None else 'None')
        col_ref_tol_str  = (self.col_ref_tol  if self.col_ref_tol  is not None else 'None')
        
        col_max_err_str  = ('{:.4E}'.format(self.col_max_err) if self.col_max_err is not None else 'None')
        
        ref_cell_str      = (self.ref_cell      if self.ref_cell      is not None else 'None')
        cell_ref_form_str = (self.cell_ref_form if self.cell_ref_form is not None else 'None')
        cell_ref_kind_str = (self.cell_ref_kind if self.cell_ref_kind is not None else 'None')
        cell_ref_tol_str  = (self.cell_ref_tol  if self.cell_ref_tol  is not None else 'None')
        
        cell_max_err_str   = ('{:.4E}'.format(self.cell_max_err) if self.cell_max_err is not None else 'None')
        
        msg = ( 'Columns: {}\n'.format(cols_str) +
                'Column Refinement Information:\n' +
                '  ref_col:      {}\n'.format(ref_col_str) + 
                '  col_ref_form: {}\n'.format(col_ref_form_str) +
                '  col_ref_kind: {}\n'.format(col_ref_kind_str) +
                '  col_ref_tol:  {}\n'.format(col_ref_tol_str) +
                '  col_max_err:  {}\n'.format(col_max_err_str) +
                'Cell Refinement Information:\n' +
                '  ref_cell:      {}\n'.format(ref_cell_str) + 
                '  cell_ref_form: {}\n'.format(cell_ref_form_str) +
                '  cell_ref_kind: {}\n'.format(cell_ref_kind_str) +
                '  cell_ref_tol:  {}\n'.format(cell_ref_tol_str) +
                '  cell_max_err:  {}\n'.format(cell_max_err_str)
               )
        
        return msg
