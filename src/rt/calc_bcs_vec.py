import numpy as np
import sys

sys.path.append('../src')
from dg.matrix import get_idx_map, get_col_idxs, get_cell_idxs, get_intr_mask
from dg.projection import push_forward, pull_back
import dg.quadrature as qd

def calc_bcs_vec(mesh, bcs_dirac):
    """
    Create the global vector corresponding to the boundary conditions.

    For now, we are going to do the slow-execution, fast development route of
    having the BCs function handle having the correct values on the boundary.
    """

    [bcs, dirac] = bcs_dirac
    
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

                    # If the BCs are a dirac-delta function, we handle
                    # them differently
                    if any(dirac):
                        in_cell = [True, True, True]
                        if dirac[0] is not None:
                            xs_f  = dirac[0]
                            if (xs_f < x0) or (x1 < xs_f):
                                in_cell[0] = False
                            else:
                                xs_b = pull_back(x0, x1, xs_f)

                        if dirac[1] is not None:
                            ys_f  = dirac[1]
                            if (ys_f < y0) or (y1 < ys_f):
                                in_cell[1] = False
                            else:
                                ys_b = pull_back(y0, y1, ys_f)

                        if dirac[2] is not None:
                            ths_f = dirac[2]
                            if (ths_f < th0) or (th1 < ths_f):
                                in_cell[2] = False
                            else:
                                ths_b = pull_back(th0, th1, ths_f)
                        
                        if any(in_cell):
                            for ii in range(0, ndof_x):
                                if ((dirac[0] is not None)
                                    and in_cell[0]):
                                    phi_i = max(0.0, qd.lag_eval(xxb, ii, xs_b))
                                    x_i   = xs_f
                                else:
                                    phi_i = 1.
                                    x_i   = xxf[ii]

                                for jj in range(0, ndof_y):
                                    if ((dirac[1] is not None)
                                        and in_cell[1]):
                                        psi_j = max(0.0, qd.lag_eval(yyb, jj, ys_b))
                                        y_j   = ys_f
                                    else:
                                        psi_j = 1.
                                        y_j   = yyf[jj]

                                    for aa in range(0, ndof_th):
                                        if ((dirac[2] is not None)
                                            and in_cell[2]):
                                            xsi_a = max(0.0, qd.lag_eval(thb, aa, ths_b))
                                            th_a  = ths_f
                                        else:
                                            xsi_a = 1.
                                            th_a  = thf[aa]
                                        
                                        bcs_ija = bcs(x_i, y_j, th_a)
                                        
                                        beta_idx = beta(ii, jj, aa)
                                        
                                        bcs_cell_vec[beta_idx] = bcs_ija \
                                            * phi_i * psi_j * xsi_a

                        
                    else:
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
