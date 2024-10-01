import numpy as np

from .Error_Indicator import Error_Indicator

import dg.quadrature as qd
from dg.projection import push_forward, pull_back

def rand_err(mesh, **kwargs):
    """
    Random error indicator for randomly refining the mesh.
    """

    default_kwargs = {'ref_col'      : True,
                      'col_ref_form' : 'hp',
                      'col_ref_kind' : 'spt',
                      'col_ref_tol'  : 0.85,
                      'ref_cell'      : True,
                      'cell_ref_form' : 'hp',
                      'cell_ref_kind' : 'ang',
                      'cell_ref_tol'  : 0.85}
    kwargs = {**default_kwargs, **kwargs}
    
    err_ind = Error_Indicator(mesh, **kwargs)
    
    col_items = sorted(mesh.cols.items())
    
    # Track maximum error(s) to calculate hp-steering only where needed
    col_max_err  = 0.
    col_ref_tol  = kwargs['col_ref_tol']
    cell_max_err = 0.
    cell_ref_tol = kwargs['cell_ref_tol']
    
    ref_forms = ['h', 'p']
    rng = np.random.default_rng()

    # Calculate errors
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:                
            # Loop through cells if necessary
            if kwargs['ref_cell']:
                cell_items = sorted(col.cells.items())
                for cell_key, cell in cell_items:
                    if cell.is_lf:
                        cell_err = rng.uniform()
                        err_ind.cols[col_key].cells[cell_key].err = cell_err
                        if cell_err >= kwargs['cell_ref_tol']:
                            if kwargs['cell_ref_form'] == 'hp':
                                err_ind.cols[col_key].cells[cell_key].ref_form = rng.choice(ref_forms)
                        else:
                            err_ind.cols[col_key].cells[cell_key].ref_form = None
                            
            if kwargs['ref_col']:
                col_err = rng.uniform()
                err_ind.cols[col_key].err = col_err
                if col_err >= kwargs['col_ref_tol']:
                    if kwargs['col_ref_form'] == 'hp':
                        err_ind.cols[col_key].ref_form = rng.choice(ref_forms)
                else:
                    err_ind.cols[col_key].ref_form = None
                    
    if kwargs['ref_col']:
        err_ind.col_max_err = 1.

    if kwargs['ref_cell']:
        err_ind.cell_max_err = 1.
                    
                
    return err_ind
