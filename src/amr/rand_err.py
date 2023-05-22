import numpy as np

from .Error_Indicator import Error_Indicator

import dg.quadrature as qd
from dg.projection import push_forward, pull_back

def rand_err(mesh, kind = 'all', form = 'hp'):
    """
    Random error indicator for testing purposes.
    """
    
    col_items = sorted(mesh.cols.items())
    ncols = len(col_items)

    if kind in ['spt', 'all']:
        by_col = True
    else:
        by_col = False
    if kind in ['ang', 'all']:
        by_cell = True
    else:
        by_cell = False
        
    err_ind = Error_Indicator(mesh, by_col = by_col, by_cell = by_cell)

    forms = ['h', 'p']
    rng = np.random.default_rng()

    for col_key, col in col_items:
        if col.is_lf:
            if form == 'hp':
                ref_form = rng.choice(forms)
            else:
                ref_form = form
            if by_col:
                err_ind.cols[col_key].err_ind = rng.uniform()
                err_ind.cols[col_key].ref_form = ref_form
            
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    if form == 'hp':
                        ref_form = rng.choice(forms)
                    else:
                        ref_form = form
                    if by_cell:
                        err_ind.cols[col_key].cells[cell_key].err_ind = rng.uniform()
                        err_ind.cols[col_key].cells[cell_key].ref_form = ref_form

    return err_ind
