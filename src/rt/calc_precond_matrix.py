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

def calc_precond_matrix(mesh, kappa, sigma, Phi, **kwargs):
    return calc_precond_matrix_mpi(mesh, kappa, sigma, Phi, **kwargs)

def calc_precond_matrix_mpi(mesh, kappa, sigma, Phi, **kwargs):
    """
    Construct the block-diagonal part of the RTDG matrix, which is:
     M_extn - M_scat
    Returns the matrix and the Preconditioner for the GMRES method.
    """
    
    default_kwargs = {'verbose'      : False, # Print info while executing
                      'blocking'     : True   # Synchronize ranks before exiting
                      }
    kwargs = {**default_kwargs, **kwargs}
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    PETSc_comm = PETSc.COMM_WORLD
    comm_rank  = PETSc_comm.getRank()
    comm_size  = PETSc_comm.getSize()
    
    if kwargs['verbose']:
        t0 = perf_counter()
        msg = (
            'Constructing (Extinction - Scattering) Matrix...\n'
            )
        utils.print_msg(msg)
    
    # Share information that is stored on root process
    mesh     = MPI_comm.bcast(mesh, root = 0)
    n_global = mesh.get_ndof()
    
    # Split the problem into parts dependent on size of COMM_WORLD.
    col_keys_global = list(sorted(mesh.cols.keys()))
    col_keys_local  = np.array_split(col_keys_global, comm_size)[comm_rank].astype(np.int32)
    
    # Get the start indices for each column matrix
    col_st_idxs = {col_keys_global[0] : 0}
    col_ndofs   = {}
    
    dof_count = 0
    for cc in range(1, len(col_keys_global)):
        col_key      = col_keys_global[cc]
        prev_col_key = col_keys_global[cc - 1]
        prev_col     = mesh.cols[prev_col_key]
        if prev_col.is_lf:
            prev_col_ndof    = 0
            [ndof_x, ndof_y] = prev_col.ndofs[:]
            
            cell_items = sorted(prev_col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [ndof_th]    = cell.ndofs[:]
                    dof_count   += ndof_x * ndof_y * ndof_th
            col_st_idxs[col_key] = dof_count
    col_ndofs[col_keys_global[-1]] = n_global - dof_count
    
    # Create PETSc sparse matrix
    M_MPI = PETSc.Mat()
    M_MPI.createAIJ(size = [n_global, n_global], comm = PETSc_comm)

    # We assemble the column matrices using the block construction of scipy
    for col_key in col_keys_local:
        [col_mtx, inv_col_mtx] = calc_col_matrix(mesh, col_key, kappa,
                                                 sigma, Phi, **kwargs)
        col_st_idx   = col_st_idxs[col_key]
        (II, JJ, VV) = sp.find(col_mtx)
        nnz_local = np.size(II)
        for idx in range(0, nnz_local):
            ii = II[idx]
            jj = JJ[idx]
            vv = VV[idx]
            M_MPI[col_st_idx + ii, col_st_idx + jj] = vv
            
    # Communicate off-rank values and setup internal data structures for
    # performing parallel operations
    M_MPI.assemblyBegin()
    M_MPI.assemblyEnd()
    
    if kwargs['verbose']:
        tf = perf_counter()
        msg = (
            'Constructed (Extinction - Scattering) Matrix\n' +
            12 * ' '  + 'Time Elapsed: {:8.4f} [s]\n'.format(tf - t0)
        )
        utils.print_msg(msg)
       
    if kwargs['blocking']:
        MPI_comm.Barrier()
        
    return [M_MPI, None]

def calc_col_matrix(mesh, col_key, kappa, sigma, Phi, **kwargs):
    
    col = mesh.cols[col_key]
    if col.is_lf:
        # Get column information, quadrature weights
        [x0, y0, x1, y1] = col.pos
        [dx, dy] = [x1 - x0, y1 - y0]
        [nx, ny] = col.ndofs
        
        [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = nx, nnodes_y = ny)
        
        xxf = proj.push_forward(x0, x1, xxb).reshape([nx, 1])
        wx  = wx.reshape([nx, 1])
        yyf = proj.push_forward(y0, y1, yyb).reshape([1, ny])
        wy  = wy.reshape([1, ny])
        
        # Extinction matrix coefficient
        wx_wy_kappa_h = wx * wy * kappa(xxf, yyf)
        # Scattering matrix coefficient
        wx_wy_sigma_h = wx * wy * sigma(xxf, yyf)
        
        # Create cell indexing for constructing column matrix
        [ncells, cell_idxs] = mat.get_cell_idxs(mesh, col_key)
        cell_mtxs = [[None] * ncells for K in range(0, ncells)]
        # Column matrix is dense, and so there are
        # up to ncell**2 non-zero cell scattering matrices
        
        cell_items = sorted(col.cells.items())
        
        # _0 refers to cell K in the equations
        # _1 refers to cell K' in the equations
        # Have nested loops because column scattering matrix is dense.
        for cell_key_0, cell_0 in cell_items:
            if cell_0.is_lf:
                # Get cell information, quadrature weights
                cell_idx_0     = cell_idxs[cell_key_0]
                [th0_0, th1_0] = cell_0.pos
                dth_0          = th1_0 - th0_0
                [nth_0]        = cell_0.ndofs
                
                [_, _, _, _, thb_0, wth_0] = qd.quad_xyth(nnodes_th = nth_0)
                
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
                        alphalist = np.zeros([cell_ndof], dtype = np.int32)
                        betalist  = np.zeros([cell_ndof], dtype = np.int32)
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
                                            Phi_ra   = Phi_h[rr, aa]
                                            
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
        col_mtx = sp.bmat(cell_mtxs, format = 'csr')
        if kwargs['precondition']:
            inv_col_mtx = spla.inv(col_mtx.tocsr())
        else:
            inv_col_mtx = None
            
    else:
        [col_mtx, inv_col_mtx] = [None, None]
    
    return [col_mtx, inv_col_mtx]
