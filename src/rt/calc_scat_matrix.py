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

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag, bmat
from time import perf_counter

from dg.matrix import get_idx_map, get_col_idxs, get_cell_idxs
from dg.projection import push_forward
import dg.quadrature as qd

from utils import print_msg

def calc_scat_matrix(mesh, sigma, Phi, **kwargs):
    return calc_scat_matrix_mpi(mesh, sigma, Phi, **kwargs)
"""
def calc_scat_matrix_new(mesh, sigma, Phi, **kwargs):
    
    default_kwargs = {"verbose" : False}
    kwargs = {**default_kwargs, **kwargs}
    
    if kwargs["verbose"]:
        t0 = perf_counter()
    
    # Create column indexing for constructing global mass matrix
    [ncols, col_idxs] = get_col_idxs(mesh)
    col_mtxs = [None] * ncols # Global scattering matrix is block-diagonal, and so
                              # there are only ncol non-zero column scattering matrices
    
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            # Get column information, quadrature weights
            col_idx = col_idxs[col_key]
            [x0, y0, x1, y1] = col.pos
            [dx, dy] = [x1 - x0, y1 - y0]
            [nx, ny] = col.ndofs
            
            [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = nx,
                                                    nnodes_y = ny)
            
            xxf = push_forward(x0, x1, xxb).reshape([nx, 1])
            wx  = wx.reshape([nx, 1])
            yyf = push_forward(y0, y1, yyb).reshape([1, ny])
            wy  = wy.reshape([1, ny])
            wx_wy_sigma_h = wx * wy * sigma(xxf, yyf)
            
            
            # Create cell indexing for constructing column mass matrix
            [ncells, cell_idxs] = get_cell_idxs(mesh, col_key)
            cell_mtxs = [[None] * ncells for K in range(0, ncells)]
                            # Column matrix is block-dense, and so there are
                            # up to ncell**2 non-zero cell scattering matrices
                            
            # _0 refers to element K in the equations
            # _1 refers to element K" in the equations
            for cell_key_0, cell_0 in sorted(col.cells.items()):
                if cell_0.is_lf:
                    # Get cell information, quadrature weights
                    cell_idx_0     = cell_idxs[cell_key_0]
                    [th0_0, th1_0] = cell_0.pos
                    dth_0          = th1_0 - th0_0
                    [nth_0]        = cell_0.ndofs
                    
                    [_, _, _, _, thb_0, wth_0] = qd.quad_xyth(nnodes_th = nth_0)
                    
                    thf_0 = push_forward(th0_0, th1_0, thb_0)
                    
                    # Indexing from p, q, r to alpha
                    alpha = get_idx_map(nx, ny, nth_0)
                    
                    # Values common to equation for each entry
                    dcoeff = dx * dy * dth_0 / 8
                    
                    for cell_key_1, cell_1 in sorted(col.cells.items()):
                        if cell_1.is_lf:
                            # Get cell information, quadrature weights
                            cell_idx_1     = cell_idxs[cell_key_1]
                            [th0_1, th1_1] = cell_1.pos
                            dth_1          = th1_1 - th0_1
                            [nth_1]        = cell_1.ndofs

                            nth_d = 15
                            [_, _, _, _, thb_1, wth_1] = qd.quad_xyth(nnodes_th = nth_1)
                            [_, _, _, _, thb_d, wth_d] = qd.quad_xyth(nnodes_th = nth_d)
                            thf_d = push_forward(th0_1, th1_1, thb_d)
                            
                            Phi_h = np.zeros([nth_0, nth_d])
                            for rr in range(0, nth_0):
                                for aa_p in range(0, nth_d):
                                    Phi_h[rr, aa_p] = Phi(thf_0[rr], thf_d[aa_p])
                            
                            # List of coordinates, values for constructing cell matrices
                            cell_ndof = nth_0 * nth_1 * nx * ny
                            alphalist = np.zeros([cell_ndof], dtype = np.int32) # alpha index
                            betalist  = np.zeros([cell_ndof], dtype = np.int32) # beta index
                            vlist     = np.zeros([cell_ndof]) # Matrix entry
                            
                            # Indexing from i, j, a to beta
                            beta = get_idx_map(nx, ny, nth_1)
                            
                            # Construct cell matrix
                            idx = 0
                            for ii in range(0, nx):
                                for jj in range(0, ny):
                                    wx_wy_sigma_ij = wx_wy_sigma_h[ii, jj]
                                    if np.abs(wx_wy_sigma_ij) > 1.e-14:
                                        for rr in range(0, nth_0):
                                            wth_0_rr = wth_0[rr]
                                            for aa in range(0, nth_1):
                                                val = 0
                                                for aa_p in range(0, nth_d):
                                                    wth_d_aap = wth_d[aa_p]
                                                    Phi_rap = Phi_h[rr, aa_p]
                                                    
                                                    val += wth_d_aap * Phi_rap * qd.lag_eval(thb_d, aa_p, thb_1[aa])
                                                    
                                                val *= dcoeff * (dth_1 / 2.0) \
                                                    * wx_wy_sigma_ij * wth_0_rr
                                                
                                                if np.abs(val) > 1.e-14:
                                                    # Index of entry
                                                    alphalist[idx] = alpha(ii, jj, rr)
                                                    betalist[idx]  = beta( ii, jj, aa)
                                                    
                                                    vlist[idx] = val
                                                    idx += 1
                                        
                            cell_mtxs[cell_idx_0][cell_idx_1] = coo_matrix((vlist, (alphalist, betalist)),
                                                                           shape = (nx * ny * nth_0,
                                                                                    nx * ny * nth_1))
                            cell_mtxs[cell_idx_0][cell_idx_1].eliminate_zeros()
                            
                            # Clean up some variables
                            del alphalist, betalist, vlist, Phi_h
                            
            # Column scattering matrix is not block-diagonal
            # but we arranged the cell matrices in the proper form
            col_mtxs[col_idx] = bmat(cell_mtxs, format = "coo")

            # Clean up some variables
            del cell_mtxs

    # Global scattering matrix is block-diagonal
    # with the column matrices as the blocks
    scat_mtx = block_diag(col_mtxs, format = "csc")

    # Clean up some variables
    del col_mtxs
    
    if kwargs["verbose"]:
        tf = perf_counter()
        msg = (
            "Scattering Matrix Construction Time: {:8.4f} [s]\n".format(tf - t0)
            )
        print_msg(msg)

    return scat_mtx

def calc_scat_matrix_old(mesh, sigma, Phi, **kwargs):
    
    default_kwargs = {"verbose" : False}
    kwargs = {**default_kwargs, **kwargs}
    
    if kwargs["verbose"]:
        t0 = perf_counter()
    
    # Create column indexing for constructing global mass matrix
    [ncols, col_idxs] = get_col_idxs(mesh)
    col_mtxs = [None] * ncols # Global scattering matrix is block-diagonal, and so
                              # there are only ncol non-zero column scattering matrices
    
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            # Get column information, quadrature weights
            col_idx = col_idxs[col_key]
            [x0, y0, x1, y1] = col.pos
            [dx, dy] = [x1 - x0, y1 - y0]
            [nx, ny] = col.ndofs
            
            [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = nx,
                                                    nnodes_y = ny)
            
            xxf = push_forward(x0, x1, xxb).reshape([nx, 1])
            wx  = wx.reshape([nx, 1])
            yyf = push_forward(y0, y1, yyb).reshape([1, ny])
            wy  = wy.reshape([1, ny])
            wx_wy_sigma_h = wx * wy * sigma(xxf, yyf)
            
            
            # Create cell indexing for constructing column mass matrix
            [ncells, cell_idxs] = get_cell_idxs(mesh, col_key)
            cell_mtxs = [[None] * ncells for K in range(0, ncells)]
                            # Column matrix is block-dense, and so there are
                            # up to ncell**2 non-zero cell scattering matrices
                            
            # _0 refers to element K in the equations
            # _1 refers to element K" in the equations
            for cell_key_0, cell_0 in sorted(col.cells.items()):
                if cell_0.is_lf:
                    # Get cell information, quadrature weights
                    cell_idx_0     = cell_idxs[cell_key_0]
                    [th0_0, th1_0] = cell_0.pos
                    dth_0          = th1_0 - th0_0
                    [nth_0]        = cell_0.ndofs
                    
                    [_, _, _, _, thb_0, wth_0] = qd.quad_xyth(nnodes_th = nth_0)
                    
                    thf_0 = push_forward(th0_0, th1_0, thb_0)
                    
                    # Indexing from p, q, r to alpha
                    alpha = get_idx_map(nx, ny, nth_0)
                    
                    # Values common to equation for each entry
                    dcoeff = dx * dy * dth_0 / 8
                    
                    for cell_key_1, cell_1 in sorted(col.cells.items()):
                        if cell_1.is_lf:
                            # Get cell information, quadrature weights
                            cell_idx_1     = cell_idxs[cell_key_1]
                            [th0_1, th1_1] = cell_1.pos
                            dth_1          = th1_1 - th0_1
                            [nth_1]        = cell_1.ndofs
                            
                            [_, _, _, _, thb_1, wth_1] = qd.quad_xyth(nnodes_th = nth_1)
                            thf_1 = push_forward(th0_1, th1_1, thb_1)
                            
                            Phi_h = np.zeros([nth_0, nth_1])
                            for rr in range(0, nth_0):
                                for aa in range(0, nth_1):
                                    Phi_h[rr, aa] = Phi(thf_0[rr], thf_1[aa])
                            
                            # List of coordinates, values for constructing cell matrices
                            cell_ndof = nth_0 * nth_1 * nx * ny
                            alphalist = np.zeros([cell_ndof], dtype = np.int32) # alpha index
                            betalist  = np.zeros([cell_ndof], dtype = np.int32) # beta index
                            vlist     = np.zeros([cell_ndof]) # Matrix entry
                            
                            # Indexing from i, j, a to beta
                            beta = get_idx_map(nx, ny, nth_1)
                            
                            # Construct cell matrix
                            idx = 0
                            for ii in range(0, nx):
                                for jj in range(0, ny):
                                    wx_wy_sigma_ij = wx_wy_sigma_h[ii, jj]
                                    if np.abs(wx_wy_sigma_ij) > 1.e-14:
                                        for rr in range(0, nth_0):
                                            wth_0_rr = wth_0[rr]
                                            for aa in range(0, nth_1):
                                                wth_1_aa = wth_1[aa]
                                                Phi_ra = Phi_h[rr, aa]
                                                
                                                val = dcoeff * (dth_1 / 2.0) \
                                                    * wx_wy_sigma_ij * wth_0_rr \
                                                    * wth_1_aa * Phi_ra
                                                
                                                if np.abs(val) > 1.e-14:
                                                    # Index of entry
                                                    alphalist[idx] = alpha(ii, jj, rr)
                                                    betalist[idx]  = beta( ii, jj, aa)
                                                    
                                                    vlist[idx] = val
                                                    idx += 1
                                        
                            cell_mtxs[cell_idx_0][cell_idx_1] = coo_matrix((vlist, (alphalist, betalist)),
                                                                           shape = (nx * ny * nth_0,
                                                                                    nx * ny * nth_1))
                            cell_mtxs[cell_idx_0][cell_idx_1].eliminate_zeros()
                            
                            # Clean up some variables
                            del alphalist, betalist, vlist, Phi_h
                            
            # Column scattering matrix is not block-diagonal
            # but we arranged the cell matrices in the proper form
            col_mtxs[col_idx] = bmat(cell_mtxs, format = "coo")

            # Clean up some variables
            del cell_mtxs

    # Global scattering matrix is block-diagonal
    # with the column matrices as the blocks
    scat_mtx = block_diag(col_mtxs, format = "csc")

    # Clean up some variables
    del col_mtxs
    
    if kwargs["verbose"]:
        tf = perf_counter()
        msg = (
            "Scattering Matrix Construction Time: {:8.4f} [s]\n".format(tf - t0)
            )
        print_msg(msg)

    return scat_mtx
"""
def calc_scat_matrix_mpi(mesh, sigma, Phi, **kwargs):
    """
    Construct the block-diagonal part of the RTDG matrix, which is:
     M_extn - M_scat
    Returns the matrix and the Preconditioner for the GMRES method.
    """
    
    default_kwargs = {"verbose"      : False, # Print info while executing
                      "blocking"     : True   # Synchronize ranks before exiting
                      }
    kwargs = {**default_kwargs, **kwargs}
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    PETSc_comm = PETSc.COMM_WORLD
    comm_rank  = PETSc_comm.getRank()
    comm_size  = PETSc_comm.getSize()
    
    if kwargs["verbose"]:
        t0 = perf_counter()
        msg = (
            "Constructing (Extinction - Scattering) Matrix...\n"
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
        [col_mtx, _] = calc_col_matrix(mesh, col_key,
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
    
    if kwargs["verbose"]:
        tf = perf_counter()
        msg = (
            "Constructed (Extinction - Scattering) Matrix\n" +
            12 * " "  + "Time Elapsed: {:8.4f} [s]\n".format(tf - t0)
        )
        utils.print_msg(msg)
       
    if kwargs["blocking"]:
        MPI_comm.Barrier()
        
    return M_MPI

def calc_col_matrix(mesh, col_key, sigma, Phi, **kwargs):
    
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
        
        # Scattering matrix coefficient
        wx_wy_sigma_h = wx * wy * sigma(xxf, yyf)
        
        # Create cell indexing for constructing column matrix
        [ncells, cell_idxs] = mat.get_cell_idxs(mesh, col_key)
        cell_mtxs = [[None] * ncells for K in range(0, ncells)]
        # Column matrix is dense, and so there are
        # up to ncell**2 non-zero cell scattering matrices
        
        cell_items = sorted(col.cells.items())
        
        # _0 refers to cell K in the equations
        # _1 refers to cell K" in the equations
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
                        
                        # Construct cell matrix - 2 cases: K = K", K != K"
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
                                    wx_wy_sigma_ij = wx_wy_sigma_h[ii, jj]
                                    for rr in range(0, nth_0):
                                        wth_0_rr = wth_0[rr]
                                        for aa in range(0, nth_1):
                                            wth_1_aa = wth_1[aa]
                                            Phi_ra   = Phi_h[rr, aa]
                                            
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
        col_mtx = sp.bmat(cell_mtxs, format = "csr")
        if kwargs["precondition"]:
            inv_col_mtx = spla.inv(col_mtx.tocsr())
        else:
            inv_col_mtx = None
            
    else:
        [col_mtx, inv_col_mtx] = [None, None]
    
    return [col_mtx, inv_col_mtx]
