import numpy as np
import sys

sys.path.append('../../src')
import dg.quadrature as qd
from rad_amr import push_forward, get_col_idxs, get_cell_idxs, get_idx_map

def get_projection_vec(mesh, f):
    """
    Create the global vector corresponding to the projection of some
    function f.
    """
    
    # Create column indexing for constructing global forcing vector,
    # global solution vector
    col_items         = sorted(mesh.cols.items())
    [ncol, col_idxs]  = get_col_idxs(mesh)
    f_col_vecs        = [None] * ncol # Global vector is a 1-D vector
    
    # Unpack f into a column vectors
    for col_key, col in col_items:
        if col.is_lf:
            # Get column information, quadrature weights
            col_idx          = col_idxs[col_key]
            [x0, y0, x1, y1] = col.pos
            [dx, dy]         = [x1 - x0, y1 - y0]
            [ndof_x, ndof_y] = col.ndofs
            
            [xxb, _, yyb, _, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                  nnodes_y = ndof_y)
            
            xxf = push_forward(x0, x1, xxb)
            yyf = push_forward(y0, y1, yyb)
            
            # Create cell indexing for constructing column forcing vector
            cell_items         = sorted(col.cells.items())
            [ncell, cell_idxs] = get_cell_idxs(col)
            f_cell_vecs        = [None] * ncell # Column forcing vector is a 1-D vector
            
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    # Get cell information, quadrature weights
                    cell_idx   = cell_idxs[cell_key]
                    [th0, th1] = cell.pos
                    dth        = th1 - th0
                    [ndof_th]  = cell.ndofs
                    
                    beta = get_idx_map(ndof_x, ndof_y, ndof_th)
                    
                    [_, _, _, _, thb, _] = qd.quad_xyth(nnodes_th = ndof_th)
                    
                    thf    = push_forward(th0, th1, thb)
                    
                    dcoeff = dx * dy * dth / 8.
                    
                    # List of entries, values for constructing the cell mask
                    cell_ndof  = ndof_x * ndof_y * ndof_th
                    f_cell_vec = np.zeros([cell_ndof])
                    for ii in range(0, ndof_x):
                        for jj in range(0, ndof_y):
                            for aa in range(0, ndof_th):
                                f_ija = f(xxf[ii], yyf[jj], thf[aa])
                                
                                beta_idx = beta(ii, jj, aa)
                                
                                f_cell_vec[beta_idx] = f_ija
                                
                    f_cell_vecs[cell_idx] = f_cell_vec
                    
            f_col_vecs[col_idx] = np.concatenate(f_cell_vecs, axis = None)
            
    f_vec = np.concatenate(f_col_vecs, axis = None)
    
    return f_vec
