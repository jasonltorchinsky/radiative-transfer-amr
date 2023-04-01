import numpy as np
import sys

sys.path.append('../src')
from dg.matrix import get_idx_map, get_col_idxs, get_cell_idxs, get_intr_mask
from dg.projection import push_forward
import dg.quadrature as qd

def calc_bcs_vec(mesh, bcs):
    """
    Create the global vector corresponding to the boundary conditions.

    For now, we are going to do the slow-execution, fast development route of
    having the BCs function handle having the correct values on the boundary.
    """

    intr_mask = get_intr_mask(mesh)
    bdry_mask = np.invert(intr_mask)
    
    # Create column indexing for constructing global bcs vector
    col_items         = sorted(mesh.cols.items())
    [ncol, col_idxs]  = get_col_idxs(mesh)
    bcs_col_vecs      = [None] * ncol # Global vector is a 1-D vector
    
    # Unpack f into a column vectors
    for col_key, col in col_items:
        if col.is_lf:
            # Get column information, quadrature weights
            col_idx          = col_idxs[col_key]
            [x0, y0, x1, y1] = col.pos
            [ndof_x, ndof_y] = col.ndofs
            
            [xxb, _, yyb, _, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                      nnodes_y = ndof_y)
            
            xxf = push_forward(x0, x1, xxb)
            yyf = push_forward(y0, y1, yyb)
            
            # Create cell indexing for constructing column bcs vector
            cell_items         = sorted(col.cells.items())
            [ncell, cell_idxs] = get_cell_idxs(mesh, col_key)
            bcs_cell_vecs      = [None] * ncell # Column forcing vector is a 1-D vector
            
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    # Get cell information, quadrature weights
                    cell_idx   = cell_idxs[cell_key]
                    [th0, th1] = cell.pos
                    [ndof_th]  = cell.ndofs
                    
                    beta = get_idx_map(ndof_x, ndof_y, ndof_th)
                    
                    [_, _, _, _, thb, _] = qd.quad_xyth(nnodes_th = ndof_th)
                    
                    thf  = push_forward(th0, th1, thb)
                    
                    # List of entries, values for constructing the cell mask
                    cell_ndof  = ndof_x * ndof_y * ndof_th
                    bcs_cell_vec = np.zeros([cell_ndof])
                    for ii in range(0, ndof_x):
                        for jj in range(0, ndof_y):
                            for aa in range(0, ndof_th):
                                bcs_ija = bcs(xxf[ii], yyf[jj], thf[aa])
                                
                                beta_idx = beta(ii, jj, aa)
                                
                                bcs_cell_vec[beta_idx] = bcs_ija
                                
                    bcs_cell_vecs[cell_idx] = bcs_cell_vec
                    
            bcs_col_vecs[col_idx] = np.concatenate(bcs_cell_vecs, axis = None)
            
    bcs_vec = np.concatenate(bcs_col_vecs, axis = None)
    
    return bcs_vec[bdry_mask]
