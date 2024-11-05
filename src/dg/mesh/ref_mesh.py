# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports

# Relative Imports

def ref_mesh(self, kind: str = "all", form: str = "h"):
    col_keys: list = sorted(self.cols.keys())
    if kind in ["ang", "all"]:
        for col_key in col_keys:
            self.ref_col(col_key, kind = "ang", form = form)
    
    if kind in ["spt", "all"]:
        for col_key in col_keys:
            self.ref_col(col_key, kind = "spt", form = form)
