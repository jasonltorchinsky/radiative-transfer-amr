import copy
import gc
import numpy               as np
import petsc4py
import psutil
import scipy.sparse        as sp
import scipy.sparse.linalg as spla
import sys
from   mpi4py              import MPI
from   petsc4py            import PETSc
from   time                import perf_counter

import dg.matrix     as mat
import dg.projection as proj
import dg.quadrature as qd
import utils

prev_col_mtxs = {}

def calc_precond_matrix(mesh, kappa, sigma, Phi, **kwargs):
    """
    Construct the block-diagonal part of the RTDG matrix, which is:
    M_intr_prop + M_extn - M_scat
    Returns the matrix and the Preconditioner for the GMRES method.
    """
    
    default_kwargs = {'verbose'      : False, # Print info while executing
                      'precondition' : False, # Calculate PC matrix
                      'blocking'     : True   # Synchronize ranks before exiting
                      }
    kwargs = {**default_kwargs, **kwargs}

    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    comm      = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()
    
    # If using too much memory, reset the saved matrices
    used_mem = psutil.virtual_memory()[2]
    if used_mem >= 80:
        global prev_col_mtxs
        del prev_col_mtxs
        gc.collect()
        prev_col_mtxs = {}
    
    if kwargs['verbose']:
        t0 = perf_counter()
        msg = (
            'Constructing (Extinction - Scattering) Matrix...\n'
            )
        utils.print_msg(msg)
        
    # Calculate these matrices in serial
    if comm_rank == 0:
        # Create column indexing for constructing global matrix
        [ncols, col_idxs] = mat.get_col_idxs(mesh)
        col_mtxs = [None] * ncols # Global matrix is block-diagonal, and so
        # there are only ncol non-zero matrices
        inv_col_mtxs = [None] * ncols # Invert the block column matrices for better preconditioning
        
        col_items = sorted(mesh.cols.items())
        for col_key, col in col_items:
            if col.is_lf:
                col_idx = col_idxs[col_key]
                have_prev = False
                if (col_key in prev_col_mtxs): # Already have column matrix
                    prev_col = prev_col_mtxs[col_key][0]
                    if col == prev_col:
                        col_mtxs[col_idx] = prev_col_mtxs[col_key][1]
                        have_prev = True
                        if kwargs['precondition']:
                            inv_col_mtxs[col_idx] = prev_col_mtxs[col_key][2]
                if not have_prev:
                    # Get column information, quadrature weights
                    [x0, y0, x1, y1] = col.pos
                    [dx, dy] = [x1 - x0, y1 - y0]
                    [nx, ny] = col.ndofs
                    
                    [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = nx,
                                                            nnodes_y = ny)
                    
                    xxf = proj.push_forward(x0, x1, xxb).reshape([nx, 1])
                    wx  = wx.reshape([nx, 1])
                    yyf = proj.push_forward(y0, y1, yyb).reshape([1, ny])
                    wy  = wy.reshape([1, ny])
                    
                    # Extinction matrix coefficient
                    wx_wy_kappa_h = wx * wy * kappa(xxf, yyf)
                    # Scattering matrix coefficient
                    wx_wy_sigma_h = wx * wy * sigma(xxf, yyf)
                    
                    # Create cell indexing for constructing column mass matrix
                    [ncells, cell_idxs] = mat.get_cell_idxs(mesh, col_key)
                    cell_mtxs = [[None] * ncells for K in range(0, ncells)]
                    # Column matrix is block-dense, and so there are
                    # up to ncell**2 non-zero cell scattering matrices
                    
                    cell_items = sorted(col.cells.items())
                    
                    # _0 refers to element K in the equations
                    # _1 refers to element K' in the equations
                    # Have nested loops because column scattering matrix is
                    # not diagonal.
                    for cell_key_0, cell_0 in cell_items:
                        if cell_0.is_lf:
                            # Get cell information, quadrature weights
                            cell_idx_0     = cell_idxs[cell_key_0]
                            [th0_0, th1_0] = cell_0.pos
                            dth_0          = th1_0 - th0_0
                            [nth_0]        = cell_0.ndofs
                            
                            [_, _, _, _, thb_0, wth_0] = \
                                qd.quad_xyth(nnodes_th = nth_0)
                            
                            thf_0 = proj.push_forward(th0_0, th1_0, thb_0)
                            
                            # Indexing from p, q, r to alpha
                            alpha = mat.get_idx_map(nx, ny, nth_0)
                            
                            # Values common to equation for each entry
                            dcoeff = dx * dy * dth_0 / 8
                            
                            for cell_key_1, cell_1 in cell_items:
                                if cell_1.is_lf:
                                    # Get cell information, quadrature weights
                                    cell_idx_1     = cell_idxs[cell_key_1]
                                    [th0_1, th1_1] = cell_1.pos
                                    dth_1          = th1_1 - th0_1
                                    [nth_1]        = cell_1.ndofs
                                    
                                    [_, _, _, _, thb_1, wth_1] = \
                                        qd.quad_xyth(nnodes_th = nth_1)
                                    thf_1 = proj.push_forward(th0_1, th1_1, thb_1)
                                    
                                    Phi_h = np.zeros([nth_0, nth_1])
                                    for rr in range(0, nth_0):
                                        for aa in range(0, nth_1):
                                            Phi_h[rr, aa] = \
                                                Phi(thf_0[rr], thf_1[aa])
                                            
                                    # List of coordinates, values for
                                    # constructing cell matrices
                                    cell_ndof = nth_0 * nth_1 * nx * ny
                                    alphalist = np.zeros([cell_ndof],
                                                         dtype = np.int32)
                                    betalist  = np.zeros([cell_ndof],
                                                         dtype = np.int32)
                                    vlist     = np.zeros([cell_ndof])
                                    
                                    # Indexing from i, j, a to beta
                                    beta = mat.get_idx_map(nx, ny, nth_1)
                                    
                                    # Construct cell matrix - 2 cases: K = K', K != K'
                                    idx = 0
                                    if cell_key_0 != cell_key_1:
                                        for ii in range(0, nx):
                                            for jj in range(0, ny):
                                                wx_wy_sigma_ij = wx_wy_sigma_h[ii, jj]
                                                if np.abs(wx_wy_sigma_ij) > 1.e-14:
                                                    for rr in range(0, nth_0):
                                                        wth_0_rr = wth_0[rr]
                                                        for aa in range(0, nth_1):
                                                            wth_1_aa = wth_1[aa]
                                                            Phi_ra = Phi_h[rr, aa]
                                                            
                                                            val = -dcoeff * (dth_1 / 2.0) \
                                                                * wx_wy_sigma_ij * wth_0_rr \
                                                                * wth_1_aa * Phi_ra
                                                            
                                                            if np.abs(val) > 1.e-14:
                                                                # Index of entry
                                                                alphalist[idx] = alpha(ii, jj, rr)
                                                                betalist[idx]  = beta( ii, jj, aa)
                                                                
                                                                vlist[idx] = val
                                                                idx += 1
                                    else: # cell_key_0 == cell_key_1
                                        for ii in range(0, nx):
                                            for jj in range(0, ny):
                                                wx_wy_kappa_ij = wx_wy_kappa_h[ii, jj]
                                                wx_wy_sigma_ij = wx_wy_sigma_h[ii, jj]
                                                for rr in range(0, nth_0):
                                                    wth_0_rr = wth_0[rr]
                                                    for aa in range(0, nth_1):
                                                        wth_1_aa = wth_1[aa]
                                                        Phi_ra = Phi_h[rr, aa]
                                                        
                                                        if rr == aa:
                                                            val = dcoeff * (wth_1_aa * wx_wy_kappa_ij # Extinction term
                                                                            - (dth_1 / 2.0) * wx_wy_sigma_ij * wth_0_rr * wth_1_aa * Phi_ra) # Scattering term
                                                        else:
                                                            val = -dcoeff * (dth_1 / 2.0) * wx_wy_sigma_ij * wth_0_rr * wth_1_aa * Phi_ra # Scattering term
                                                            
                                                        if np.abs(val) > 1.e-14:
                                                            # Index of entry
                                                            alphalist[idx] = alpha(ii, jj, rr)
                                                            betalist[idx]  = beta( ii, jj, aa)
                                                            
                                                            vlist[idx] = val
                                                            idx += 1
                                                            
                                    cell_mtxs[cell_idx_0][cell_idx_1] = \
                                        sp.coo_matrix((vlist, (alphalist, betalist)),
                                                   shape = (nx * ny * nth_0,
                                                            nx * ny * nth_1))
                                    cell_mtxs[cell_idx_0][cell_idx_1].eliminate_zeros()
                                    
                    # Column scattering matrix is not block-diagonal
                    # but we arranged the cell matrices in the proper form
                    col_mtx = sp.bmat(cell_mtxs, format = 'coo')
                    col_mtxs[col_idx] = col_mtx
                    if kwargs['precondition']:
                        inv_col_mtx = spla.inv(col_mtx.tocsr()).tocoo()
                        prev_col_mtxs[col_key] = [deepcopy(col), col_mtx, inv_col_mtx]
                        inv_col_mtxs[col_idx] = inv_col_mtx
                        
        # Global matrix is block-diagonal
        # with the column matrices as the blocks
        mtx = sp.block_diag(col_mtxs, format = 'csr')
        if kwargs['precondition']:
            pre_mtx = sp.block_diag(inv_col_mtxs, format = 'csr')
        else:
            pre_mtx = None
            
    else:
       [mtx, pre_mtx] = [0, 0]
    
    if kwargs['verbose']:
        tf = perf_counter()
        msg = (
            'Constructed (Extinction - Scattering) Matrix\n' +
            12 * ' '  + 'Time Elapsed: {:8.4f} [s]\n'.format(tf - t0)
        )
        utils.print_msg(msg)
       
    if kwargs['blocking']:
        MPI_comm.Barrier()
        
    return [mtx, pre_mtx]

