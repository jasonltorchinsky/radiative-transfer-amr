class Error_Indicator_Column:
    def __init__(self, **kwargs):

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
        self.ref_col  = kwargs['ref_col']
        if self.ref_col:
            self.ref_form = kwargs['col_ref_form']
            self.ref_kind = kwargs['col_ref_kind']
            self.ref_tol  = kwargs['col_ref_tol']
            self.err      = 0.
        else: # If we're not refining columns, ignore column refinement paremeters
            self.ref_form = None
            self.ref_kind = None
            self.ref_tol  = None
            self.err      = None
        
        # Do we refine cells? If so, how?
        self.ref_cell      = kwargs['ref_cell']
        self.cell_ref_form = kwargs['cell_ref_form']
        self.cell_ref_kind = kwargs['cell_ref_kind']
        self.cell_ref_tol  = kwargs['cell_ref_tol']
        
        if self.ref_cell:
            self.cells = {}
        else:
            self.cells = None

    def __str__(self):
        msg = ( 'Column Refinement Information:\n' +
                '  ref_col:  {}\n'.format(self.ref_col) + 
                '  ref_form: {}\n'.format(self.ref_form) +
                '  ref_kind: {}\n'.format(self.ref_kind) +
                '  ref_tol:  {}\n'.format(self.ref_tol) +
                '  err:      {:.4E}\n'.format(self.err) +
                'Cells: {}\n'.format(sorted(list(self.cells.keys()))) +
                'Cell Refinement Information:\n' +
                '  ref_cell:      {}\n'.format(self.ref_cell) + 
                '  cell_ref_form: {}\n'.format(self.cell_ref_form) +
                '  cell_ref_kind: {}\n'.format(self.cell_ref_kind) +
                '  cell_ref_tol:  {}\n'.format(self.cell_ref_tol)
               )

        return msg
