# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports

# Relative Imports


def mean_error_to_resolve(self) -> float:
    """
    Average error indicator over the elements marked for refinement, i.e.,
    self.error_to_resolve (their sum) divided by the number of marked
    columns and cells. This is the average used by the spatial-angular
    steering criterion (Eq. 16).
    """
    nmarked: int = 0
    col_items: list = sorted(self.cols.items())
    for _, col in col_items:
        if col.do_ref:
            nmarked += 1

        cell_items: list = sorted(col.cells.items())
        for _, cell in cell_items:
            if cell.do_ref:
                nmarked += 1

    return self.error_to_resolve / max(nmarked, 1)
