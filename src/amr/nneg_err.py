import numpy as np

from .Error_Indicator import Error_Indicator
from .hp_steer import hp_steer_col, hp_steer_cell

import dg.quadrature as qd
from dg.projection import push_forward, pull_back

def nneg_err(mesh, proj, **kwargs):
    """
    Refine all with a negative value
    """
    default_kwargs = {'ref_col'      : True,
                      'col_ref_form' : 'hp',
                      'col_ref_kind' : 'spt',
                      'col_ref_tol'  : 0.0,
                      'ref_cell'      : True,
                      'cell_ref_form' : 'hp',
                      'cell_ref_kind' : 'ang',
                      'cell_ref_tol'  : 0.0}
    kwargs = {**default_kwargs, **kwargs}
    
    err_ind = Error_Indicator(mesh, **kwargs)
    
    col_items = sorted(mesh.cols.items())

    # Assign error of 1. to cols, cells with a negative value
    for col_key, col in col_items:
        if col.is_lf:
            if kwargs['ref_col']:
                col_has_neg = False
                
            # Loop through cells
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    uh_cell = proj.cols[col_key].cells[cell_key].vals
                    if kwargs['ref_cell']:
                        cell_has_neg = np.any(uh_cell < kwargs['cell_ref_tol'])
                        cell_err = int(cell_has_neg)
                        
                    if kwargs['ref_col']:
                        col_has_neg = col_has_neg or np.any(uh_cell < kwargs['col_ref_tol'])
                    
                    if kwargs['ref_cell']:
                        err_ind.cols[col_key].cells[cell_key].err = cell_err
                        if cell_has_neg:
                            if kwargs['cell_ref_form'] == 'hp':
                                err_ind.cols[col_key].cells[cell_key].ref_form = hp_steer_cell(mesh, proj, col_key, cell_key)
                        else:
                            err_ind.cols[col_key].cells[cell_key].ref_form = None
                            
            if kwargs['ref_col']:
                err_ind.cols[col_key].err = int(col_has_neg)
                if col_has_neg:
                    if kwargs['col_ref_form'] == 'hp':
                        err_ind.cols[col_key].ref_form = hp_steer_col(mesh, proj, col_key)
                else:
                    err_ind.cols[col_key].ref_form = None
                        
    if kwargs['ref_col']:
        err_ind.col_max_err = 1.
        
    if kwargs['ref_cell']:
        err_ind.cell_max_err = 1.
            
    return err_ind
