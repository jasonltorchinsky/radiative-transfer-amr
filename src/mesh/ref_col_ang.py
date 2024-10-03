# Refine a column, angularly
def ref_col_ang(self, col_key):
    col = self.cols[col_key]
    if col.is_lf:
        cell_keys = sorted(col.cells.keys())
        for cell_key in cell_keys:
            self.ref_cell(col_key, cell_key)
