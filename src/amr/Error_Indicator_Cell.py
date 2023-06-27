class Error_Indicator_Cell:
    def __init__(self, **kwargs):

        default_kwargs = {'ref_cell'      : False,
                          'cell_ref_form' : None,
                          'cell_ref_kind' : None,
                          'cell_ref_tol'  : 1.1}
        kwargs = {**default_kwargs, **kwargs}
        
        # Do we refine cells? If so, how?
        self.ref_cell = kwargs['ref_cell']
        if self.ref_cell:
            self.ref_form = kwargs['cell_ref_form']
            self.ref_kind = kwargs['cell_ref_kind']
            self.ref_tol  = kwargs['cell_ref_tol']
            self.err      = 0.
        else: # If we're not refining cells, ignore cell refinement parameters
            self.ref_form = None
            self.ref_kind = None
            self.ref_tol  = None
            self.err      = None
            
    def __str__(self):
        msg = ( 'Cell Refinement Information:\n' +
                '  ref_cell: {}\n'.format(self.ref_cell) + 
                '  ref_form: {}\n'.format(self.ref_form) +
                '  ref_kind: {}\n'.format(self.ref_kind) +
                '  ref_tol:  {}\n'.format(self.ref_tol) +
                '  err:      {:.4E}\n'.format(self.err)
               )
        
        return msg
