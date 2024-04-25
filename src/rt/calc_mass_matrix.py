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

def calc_mass_matrix(mesh, kappa, **kwargs):
    return calc_mass_matrix_mpi(mesh, kappa, **kwargs)
"""
def calc_mass_matrix_seq(mesh, kappa, **kwargs):

    default_kwargs = {'verbose' : False}
    kwargs = {**default_kwargs, **kwargs}
    
    if kwargs['verbose']:
        t0 = perf_counter()

    # Create column indexing for constructing global mass matrix
    [ncols, col_idxs] = get_col_idxs(mesh)
    col_mtxs = [None] * ncols # Global mass matrix is block-diagonal, and so
                              # there are only ncol non-zero column mass matrices

    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            # Get column information, quadrature weights
            col_idx = col_idxs[col_key]
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy] = [x1 - x0, y1 - y0]
            [ndof_x, ndof_y] = col.ndofs[:]
            
            [xxb, w_x, yyb, w_y, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                      nnodes_y = ndof_y)
            
            xxf = push_forward(x0, x1, xxb).reshape([ndof_x, 1])
            yyf = push_forward(y0, y1, yyb).reshape([1, ndof_y])

            w_x = w_x.reshape([ndof_x, 1])
            w_y = w_y.reshape([1, ndof_y])
            kappa_col = kappa(xxf, yyf)
            wx_wy_kappa_col = w_x * w_y * kappa_col
            
            # Create cell indexing for constructing column mass matrix
            [ncells, cell_idxs] = get_cell_idxs(mesh, col_key)
            cell_mtxs = [None] * ncells # Column mass matrix is block-diagonal, and
                                        # so there are only ncell non-zero cell
                                        # mass matrices

            for cell_key, cell in sorted(col.cells.items()):
                if cell.is_lf:
                    # Get cell information, quadrature weights
                    cell_idx   = cell_idxs[cell_key]
                    [th0, th1] = cell.pos
                    dth        = th1 - th0
                    [ndof_th]  = cell.ndofs
                    
                    [_, _, _, _, _, w_th] = qd.quad_xyth(nnodes_th = ndof_th)
                    
                    cell_ndof = ndof_x * ndof_y * ndof_th
                    betalist  = np.zeros([cell_ndof], dtype = np.int32) # beta index
                    vlist     = np.zeros([cell_ndof]) # Entry value

                    # Indexing from i, j, a to beta
                    # In this case, the alpha and beta indices are the same,
                    # so we don't have to do them separately
                    beta = get_idx_map(ndof_x, ndof_y, ndof_th)

                    # Values common to equation for each entry
                    dcoeff = dx * dy * dth / 8

                    # Construct cell matrix
                    idx = 0
                    for ii in range(0, ndof_x):
                        for jj in range(0, ndof_y):
                            wx_wy_kappa_ij = wx_wy_kappa_col[ii, jj]
                            for aa in range(0, ndof_th):
                                wth_a = w_th[aa]
                                
                                # Calculate entry index, value
                                betalist[idx] = beta(ii, jj, aa)
                                
                                vlist[idx] = dcoeff * wth_a * wx_wy_kappa_ij
                                
                                idx += 1
                                
                    cell_mtxs[cell_idx] = coo_matrix((vlist, (betalist, betalist)))
                    
            col_mtxs[col_idx] = block_diag(cell_mtxs, format = 'coo')

    # Global mass matrix is block-diagonal
    # with the column matrices as the blocks
    mass_mtx = block_diag(col_mtxs, format = 'csc')
    
    if kwargs['verbose']:
        tf = perf_counter()
        msg = (
            'Mass Matrix Construction Time: {:8.4f} [s]\n'.format(tf - t0)
            )
        print_msg(msg)

    return mass_mtx
"""
    
def calc_mass_matrix_mpi(mesh, kappa, **kwargs):

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
            'Constructing Interior Propagation Matrix...\n'
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
                    [ndof_th]  = cell.ndofs[:]
                    dof_count += ndof_x * ndof_y * ndof_th
            col_st_idxs[col_key]    = dof_count
    col_ndofs[col_keys_global[-1]] = n_global - dof_count
    
    # Create PETSc sparse matrix
    M_MPI = PETSc.Mat()
    M_MPI.createAIJ(size = [n_global, n_global], comm = PETSc_comm)

    # We assemble the column matrices using the block construction of scipy
    for col_key in col_keys_local:
        col_mtx      = calc_col_matrix(mesh, col_key, kappa, **kwargs)
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
            'Constructed Interior Propagation Matrix\n' +
            12 * ' '  + 'Time Elapsed: {:8.4f} [s]\n'.format(tf - t0)
        )
        utils.print_msg(msg)
       
    if kwargs['blocking']:
        MPI_comm.Barrier()
        
    return M_MPI

def calc_col_matrix(mesh, col_key, kappa, **kwargs):
        
    col = mesh.cols[col_key]
    if col.is_lf:
        # Get column information, quadrature weights
        [x0, y0, x1, y1] = col.pos
        [dx, dy] = [x1 - x0, y1 - y0]
        [nx, ny] = col.ndofs
        
        [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = nx, nnodes_y = ny)

        xxf = proj.push_forward(x0, x1, xxb).reshape([nx, 1])
        yyf = proj.push_forward(y0, y1, yyb).reshape([1, ny])
        wx = wx.reshape([nx, 1])
        wy = wy.reshape([1, ny])
        
        kappa_col = kappa(xxf, yyf)
        wx_wy_kappa_col = wx * wy * kappa_col
        
        # Create cell indexing for constructing column matrix
        [ncells, cell_idxs] = proj.get_cell_idxs(mesh, col_key)
        cell_mtxs = [None] * ncells # Column interior convection  matrix is
        # block-diagonal, and so there are only 
        # ncell non-zero cell interior convection
        # matrices
        
        for cell_key, cell in sorted(col.cells.items()):
            if cell.is_lf:
                # Get cell information, quadrature weights
                cell_idx   = cell_idxs[cell_key]
                [th0, th1] = cell.pos
                dth        = th1 - th0
                [nth]      = cell.ndofs[:]
                [_, _, _, _, _, w_th] = qd.quad_xyth(nnodes_th = nth)
                
                dcoeff = dx * dy * dth / 8.
                
                # Indexing from i, j, a to beta
                # Same formula for p, q, r to alpha, but we define alpha
                # anyway for clarity
                alpha = mat.get_idx_map(nx, ny, nth)
                beta  = mat.get_idx_map(nx, ny, nth)
                
                # Set up arrays for delta_ip * delta_ar term
                cell_ndof = nx * ny * nth
                betalist  = np.zeros([cell_ndof], dtype = np.int32) # beta index
                vlist     = np.zeros([cell_ndof]) # Entry value
                
                # Construct cell matrix
                idx = 0
                for ii in range(0, nx):
                    for jj in range(0, ny):
                        wx_wy_kappa_ij = wx_wy_kappa_col[ii, jj]
                        for aa in range(0, nth):
                            wth_a = w_th[aa]
                            
                            # Calculate entry index, value
                            betalist[idx] = beta(ii, jj, aa)
                            
                            vlist[idx] = dcoeff * wth_a * wx_wy_kappa_ij
                            
                            idx += 1
                            
                cell_mtxs[cell_idx] = sp.coo_matrix((vlist, (betalist, betalist)))
                
        col_mtx = sp.block_diag(cell_mtxs, format = 'csr')
        
    else:
        col_mtx = None
        
    return col_mtx
