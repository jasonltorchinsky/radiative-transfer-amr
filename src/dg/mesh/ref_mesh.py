# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports

# Relative Imports

def ref_mesh(self, kind: str = "all", form: str = "h"):
    col_keys = sorted(self.cols.keys())
    if (kind == "ang") or (kind == "all"):
        for col_key in col_keys:
            self.ref_col(col_key, kind = "ang", form = form)
    
    if (kind == "spt") or (kind == "all"):
        for col_key in col_keys:
            self.ref_col(col_key, kind = "spt", form = form)
