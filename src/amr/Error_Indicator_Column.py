class Error_Indicator_Column:
    def __init__(self, ref_form = None, ref_kind = None,
                 tol_spt = 1.1,   tol_ang = 1.1,
                 by_col  = False, by_cell = False):
        
        self.ref_form = ref_form
        self.ref_kind = ref_kind
        self.tol_spt  = tol_spt
        self.tol_ang  = tol_ang
        self.by_col   = by_col
        self.by_cell  = by_cell

        self.max_err = 0.
        self.err     = 0.
        self.cells   = {}

    def __str__(self):
        msg = ( 'Hello World!'
               )

        return msg
