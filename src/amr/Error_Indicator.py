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
        else:
            self.col_max_err = None
            
        # Do we refine cells? If so, how?
        self.ref_cell      = kwargs['ref_cell']
        self.cell_ref_form = kwargs['cell_ref_form']
        self.cell_ref_kind = kwargs['cell_ref_kind']
        self.cell_ref_tol  = kwargs['cell_ref_tol']
        
        if self.ref_cell:
            self.cell_max_err = 0.
        else:
            self.cell_max_err = None
        
        
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
        msg = ( 'Columns: {}\n'.format(sorted(list(self.cols.keys()))) +
                'Column Refinement Information:\n' +
                '  ref_col:      {}\n'.format(self.ref_col) + 
                '  col_ref_form: {}\n'.format(self.col_ref_form) +
                '  col_ref_kind: {}\n'.format(self.col_ref_kind) +
                '  col_ref_tol:  {}\n'.format(self.col_ref_tol) +
                '  col_max_err:  {:.4E}\n'.format(self.col_max_err) +
                'Cell Refinement Information:\n' +
                '  ref_cell:      {}\n'.format(self.ref_cell) + 
                '  cell_ref_form: {}\n'.format(self.cell_ref_form) +
                '  cell_ref_kind: {}\n'.format(self.cell_ref_kind) +
                '  cell_ref_tol:  {}\n'.format(self.cell_ref_tol) +
                '  cell_max_err:  {:.4E}\n'.format(self.cell_max_err)
               )
        
        return msg
