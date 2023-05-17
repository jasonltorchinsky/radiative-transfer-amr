class Error_Indicator_Cell:
    def __init__(self, by_col, by_cell):
        if by_cell:
            self.err_ind = 0.
            self.ref_form = ''
        else:
            self.err_ind = None
            self.ref_form = None
        
        

    def __str__(self):
        msg = ( 'Hello World!'
               )

        return msg
