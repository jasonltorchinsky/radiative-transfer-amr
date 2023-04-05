import numpy as np

def ref_by_ind(mesh, err_ind, ref_ratio):
    """
    Refine the mesh by some error indicator (err_ind).
    """

    # !!!We're going to assume the error indicator is by column for now.!!!
    max_err = get_max_err_ind(err_ind, by_col = True, by_cell = False)

    col_keys = sorted(mesh.cols.keys())
    for col_key in col_keys:
        if col_key in mesh.cols.keys(): # There's a chance we refine a column
            # and delete it from the mesh
            col = mesh.cols[col_key]
            if col.is_lf:
                col_err_ind = err_ind.cols[col_key].err_ind
                if col_err_ind >= ref_ratio * max_err:
                    mesh.ref_col(col_key, kind = 'spt')    

    return mesh

def get_max_err_ind(err_ind, by_col, by_cell):

    max_err_ind = 0.
    if err_ind.by_col and by_col:
        col_keys = sorted(err_ind.cols.keys())
        for col_key in col_keys:
            err_ind_col = err_ind.cols[col_key]
            max_err_ind = max([max_err_ind, err_ind_col.err_ind])

    if err_ind.by_cell and by_cell:
        col_keys = sorted(err_ind.cols.keys())
        for col_key in col_keys:
            err_ind_col = err_ind.cols[col_key]
            cell_keys = sorted(err_ind_col.cells.keys())
            for cell_key in cell_keys:
                err_ind_cell = err_ind_col.cells[cell_key]
                max_err_ind = max([max_err_ind, err_ind_cell.err_ind])

    return max_err_ind
