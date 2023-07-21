import numpy    as     np
import petsc4py
from   mpi4py   import MPI
from   petsc4py import PETSc
from   time     import perf_counter

import dg.matrix     as mat
import dg.projection as proj
import dg.quadrature as qd
import utils

def calc_forcing_vec(mesh, f, **kwargs):
    """
    Create the global vector corresponding to the forcing term f.
    """

    default_kwargs = {'precondition' : False, # Calculate PC matrix
                      'verbose'      : False, # Print info while executing
                      'blocking'     : True   # Synchronize ranks before exiting
                      } 
    kwargs = {**default_kwargs, **kwargs}
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    comm      = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()

    if kwargs['verbose']:
        t0 = perf_counter()
        msg = (
            'Constructing Forcing Vector...\n'
            )
        utils.print_msg(msg)
    
    if comm_rank == 0:
        # Create column indexing for constructing global forcing vector,
        # global solution vector
        col_items         = sorted(mesh.cols.items())
        [ncol, col_idxs]  = mat.get_col_idxs(mesh)
        f_col_vecs        = [None] * ncol # Global vector is a 1-D vector
        
        # Unpack f into a column vectors
        for col_key, col in col_items:
            if col.is_lf:
                # Get column information, quadrature weights
                col_idx          = col_idxs[col_key]
                [x0, y0, x1, y1] = col.pos
                [dx, dy]         = [x1 - x0, y1 - y0]
                [ndof_x, ndof_y] = col.ndofs
                
                [xxb, w_x, yyb, w_y, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                          nnodes_y = ndof_y)
                
                xxf = proj.push_forward(x0, x1, xxb)
                yyf = proj.push_forward(y0, y1, yyb)
                
                # Create cell indexing for constructing column forcing vector
                cell_items         = sorted(col.cells.items())
                [ncell, cell_idxs] = mat.get_cell_idxs(mesh, col_key)
                f_cell_vecs        = [None] * ncell
                # Column forcing vector is a 1-D vector
                
                for cell_key, cell in cell_items:
                    if cell.is_lf:
                        # Get cell information, quadrature weights
                        cell_idx   = cell_idxs[cell_key]
                        [th0, th1] = cell.pos
                        dth        = th1 - th0
                        [ndof_th]  = cell.ndofs
                        
                        beta = mat.get_idx_map(ndof_x, ndof_y, ndof_th)
                        
                        [_, _, _, _, thb, w_th] = qd.quad_xyth(nnodes_th = ndof_th)
                        
                        thf = proj.push_forward(th0, th1, thb)
                        
                        dcoeff = dx * dy * dth / 8.
                        
                        # List of entries, values for constructing the cell mask
                        cell_ndof  = ndof_x * ndof_y * ndof_th
                        f_cell_vec = np.zeros([cell_ndof])
                        for ii in range(0, ndof_x):
                            wx_i = w_x[ii]
                            for jj in range(0, ndof_y):
                                wy_j = w_y[jj]
                                for aa in range(0, ndof_th):
                                    wth_a = w_th[aa]
                                    f_ija = f(xxf[ii], yyf[jj], thf[aa])
                                    
                                    beta_idx = beta(ii, jj, aa)
                                    
                                    f_cell_vec[beta_idx] = dcoeff * wx_i * wy_j \
                                        * wth_a * f_ija
                                    
                        f_cell_vecs[cell_idx] = f_cell_vec
                        
                f_col_vecs[col_idx] = np.concatenate(f_cell_vecs, axis = None)
                
        f_vec = np.concatenate(f_col_vecs, axis = None)
    else:
        f_vec = np.zeros([1])
        
    if kwargs['verbose']:
        tf = perf_counter()
        msg = (
            'Constructed Forcing Vector\n' +
            12 * ' '  + 'Time Elapsed: {:8.4f} [s]\n'.format(tf - t0)
        )
        utils.print_msg(msg)
    
    if kwargs['blocking']:        
        MPI_comm.Barrier()
    
    return f_vec
