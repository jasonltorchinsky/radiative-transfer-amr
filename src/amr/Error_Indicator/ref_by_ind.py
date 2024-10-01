def ref_by_ind(mesh, err_ind):
    """
    Refine the mesh by some error indicator (err_ind).
    """
    
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            if col_key in err_ind.cols.keys(): # Avoid trying to refine columns
                # that have been refined from 1-irregularity
                # First check if cells need to be refined, and refine them
                if err_ind.ref_cell:
                    cell_items = sorted(col.cells.items())
                    for cell_key, cell in cell_items:
                        if cell.is_lf:
                            if cell_key in err_ind.cols[col_key].cells.keys(): # Avoid trying to refine cells
                                # that have been refined from 1-irregularity
                                ref_form = err_ind.cols[col_key].cells[cell_key].ref_form
                                if ref_form:
                                    mesh.ref_cell(col_key, cell_key, form = ref_form)
                            
                if err_ind.ref_col:
                    ref_form = err_ind.cols[col_key].ref_form
                    ref_kind = err_ind.cols[col_key].ref_kind
                    if ref_form and ref_kind:
                        mesh.ref_col(col_key, kind = ref_kind, form = ref_form)
                        
    return mesh
