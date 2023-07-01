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
        ref_cell_str  = (self.ref_cell  if self.ref_cell is not None else 'None')
        ref_form_str  = (self.ref_form  if self.ref_form is not None else 'None')
        ref_kind_str  = (self.ref_kind  if self.ref_kind is not None else 'None')
        ref_tol_str   = (self.ref_tol   if self.ref_tol  is not None else 'None')
        err_str       = ('{:.4E}'.format(self.err) if self.err is not None else 'None')
        
        msg = ( 'Cell Refinement Information:\n' +
                '  ref_cell: {}\n'.format(ref_cell_str) + 
                '  ref_form: {}\n'.format(ref_form_str) +
                '  ref_kind: {}\n'.format(ref_kind_str) +
                '  ref_tol:  {}\n'.format(ref_tol_str) +
                '  err:      {}\n'.format(err_str)
               )
        
        return msg
