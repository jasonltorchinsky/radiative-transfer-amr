class Error_Indicator_Column:
    def __init__(self, by_col, by_cell):
        if by_col:
            self.err_ind = 0.
        else:
            self.err_ind = None
            
        if by_cell:
            self.cells = {}
        else:
            self.cells = None

    def __str__(self):
        msg = ( 'Hello World!'
               )

        return msg