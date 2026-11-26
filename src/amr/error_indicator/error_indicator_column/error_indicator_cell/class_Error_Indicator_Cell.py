# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports

# Relative Imports

class Error_Indicator_Cell:
    def __init__(self, error: float, ref_form: str = "", do_ref: bool = False):
        self.error: float = error
        self.ref_form: str = ref_form
        self.do_ref: bool = do_ref
