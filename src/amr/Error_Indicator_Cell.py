class Error_Indicator_Cell:
    def __init__(self, by_col, by_cell):
        if by_col:
            self.err_ind = 0.
        else:
            self.err_ind = None
        
        

    def __str__(self):
        msg = ( 'Hello World!'
               )

        return msg
