import numpy as np

def ref_by_ind(mesh, err_ind, ref_ratio):
    """
    Refine the mesh by some error indicator (err_ind).
    """

    by_col  = err_ind.by_col
    by_cell = err_ind.by_cell
    
    [max_col_err, max_cell_err] = get_max_err_ind(err_ind,
                                                  by_col  = by_col,
                                                  by_cell = by_cell)
    
    # This could be more efficient for sure, this is first pass.
    if by_cell:
        col_keys = sorted(mesh.cols.keys())
        for col_key in col_keys:
            col = mesh.cols[col_key]
            cell_keys = sorted(col.cells.keys())
            for cell_key in cell_keys:
                mesh_curr_cell_keys = sorted(col.cells.keys())
                err_ind_curr_cell_keys = sorted(err_ind.cols[col_key].cells.keys())
                if ((cell_key in mesh_curr_cell_keys) and
                    (cell_key in err_ind_curr_cell_keys)):
                    cell = col.cells[cell_key]
                    if cell.is_lf:
                        cell_err_ind = err_ind.cols[col_key].cells[cell_key].err_ind
                        if cell_err_ind >= ref_ratio * max_cell_err:
                            mesh.ref_cell(col_key, cell_key)
                            
    if by_col:
        col_keys = sorted(mesh.cols.keys())
        for col_key in col_keys:
            if col_key in mesh.cols.keys(): # There's a chance we refine a column
                # and delete it from the mesh
                col = mesh.cols[col_key]
                if col.is_lf:
                    col_err_ind = err_ind.cols[col_key].err_ind
                    if col_err_ind >= ref_ratio * max_col_err:
                        mesh.ref_col(col_key, kind = 'spt')
    
    return mesh

def get_max_err_ind(err_ind, by_col, by_cell):

    max_col_err_ind = -1.
    max_cell_err_ind = -1.
    if err_ind.by_col and by_col:
        col_keys = sorted(err_ind.cols.keys())
        for col_key in col_keys:
            err_ind_col = err_ind.cols[col_key]
            max_col_err_ind = max([max_col_err_ind, err_ind_col.err_ind])

    if err_ind.by_cell and by_cell:
        col_keys = sorted(err_ind.cols.keys())
        for col_key in col_keys:
            err_ind_col = err_ind.cols[col_key]
            cell_keys = sorted(err_ind_col.cells.keys())
            for cell_key in cell_keys:
                err_ind_cell = err_ind_col.cells[cell_key]
                max_cell_err_ind = max([max_cell_err_ind, err_ind_cell.err_ind])

    if max_col_err_ind < 0.:
        max_col_err_ind = None

    if max_cell_err_ind < 0.:
        max_cell_err_ind = None
        
    return [max_col_err_ind, max_cell_err_ind]
