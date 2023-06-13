import numpy as np

from .Error_Indicator import Error_Indicator
from .hp_steer import hp_steer_col, hp_steer_cell

import dg.quadrature as qd
from dg.projection import push_forward, pull_back

def nneg_err(mesh, proj, tol = 0.0, kind = 'all', form = 'hp'):
    """
    Error indicator to refine all columns, cells with a negative value.
    """

    if kind in ['spt', 'all']:
        by_col = True
    else:
        by_col = False
    if kind in ['ang', 'all']:
        by_cell = True
    else:
        by_cell = False
    
    col_items = sorted(mesh.cols.items())
    ncols = len(col_items)
    
    err_ind = Error_Indicator(mesh, by_col = by_col, by_cell = by_cell)

    max_err = 0.
    
    for col_key, col in col_items:
        if col.is_lf:
            col_has_neg = False
            
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    uh_cell = proj.cols[col_key].cells[cell_key].vals
                    cell_has_neg = np.any(uh_cell < tol)
                    
                    cell_err = int(cell_has_neg)
                    
                    col_has_neg = cell_has_neg or col_has_neg

                    if by_cell:
                        err_ind.cols[col_key].cells[cell_key].err_ind = cell_err
                        if form == 'hp':
                            ref_form = hp_steer_cell(mesh, proj, col_key, cell_key)
                        else:
                            ref_form = form
                        err_ind.cols[col_key].cells[cell_key].ref_form = ref_form

            if by_col:
                err_ind.cols[col_key].err_ind = int(col_has_neg)
                if form == 'hp':
                    ref_form = hp_steer_col(mesh, proj, col_key)
                else:
                    ref_form = form
                err_ind.cols[col_key].ref_form = ref_form
                
            max_err = float(max(max_err, int(col_has_neg)))

    if max_err > 0.5: # If any cells are negative, mark the maximum as positive so we refine
        err_ind.max_err = max_err
    else: # No cells are negative, so we mark the maximum as negative so we don't refine
        err_ind.max_err = -1.
            
    return err_ind
