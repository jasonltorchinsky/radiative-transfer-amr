import numpy        as np
import petsc4py
import scipy.sparse as sp
import sys
from   mpi4py       import MPI
from   petsc4py     import PETSc
from   time         import perf_counter

from .get_Ex  import get_Ex
from .get_Ey  import get_Ey
from .get_Eth import get_Eth

import dg.mesh       as ji_mesh
import dg.matrix     as mat
import dg.projection as proj
import dg.quadrature as qd
import utils

def calc_bdry_conv_matrix(mesh, **kwargs):
    return calc_bdry_conv_matrix_mpi(mesh, **kwargs)

def calc_bdry_conv_matrix_seq(mesh, **kwargs):
    
    default_kwargs = {'verbose'  : False, # Print info while executing
                      'blocking' : True   # Synchronize ranks before exiting
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
            'Constructing Boundary Propagation Matrix...\n'
            )
        utils.print_msg(msg)
        
    # Calculate these matrices in serial
    if comm_rank == 0:
        # Variables that are the same throughout the loops
        col_items = sorted(mesh.cols.items())
        
        # Create column indexing for constructing global mass matrix
        [ncols, col_idxs] = proj.get_col_idxs(mesh)
        col_mtxs = [[None] * ncols for C in range(0, ncols)] # We have to assemble a
        # lot on inter-column interaction matrices,
        # so the construction is a bit more difficult.
        
        # The local-column matrices come in two kinds: M^CC and M^CC'.
        # The M^CC have to be constructed in four parts: M^CC_F.
        # The M^CC' can be constructed in one part.
        # We loop through each column C, then through each face F of C.
        # For each face, loop through each element K of C.
        # Depending on K, we contribute to M^CC_F or M^CC'.
        # Hold all four M^CC_F, add them together after all of the loops.
        for col_key_0, col_0 in col_items:
            if col_0.is_lf:
                # Use _0 to refer to column C
                # Later, use _1 to refer to column C'
                col_idx_0 = col_idxs[col_key_0]
                
                # Loop through the faces of C
                for F in range(0, 4):
                    [col_mtx_00, [col_key_1, col_mtx_01], [col_key_2, col_mtx_02]] = \
                        calc_col_matrix(mesh, col_key_0, F)
                    
                    # The intra-column matrix may already exist. If so, add to it.
                    if col_mtxs[col_idx_0][col_idx_0] is None:
                        col_mtxs[col_idx_0][col_idx_0] = col_mtx_00
                    else:                        
                        col_mtxs[col_idx_0][col_idx_0] += col_mtx_00
                        
                    if col_key_1 is not None:
                        col_idx_1 = col_idxs[col_key_1]
                        if col_mtxs[col_idx_0][col_idx_1] is None:
                            col_mtxs[col_idx_0][col_idx_1] = col_mtx_01
                        else:                        
                            col_mtxs[col_idx_0][col_idx_1] += col_mtx_01
                    if col_key_2 is not None:
                        col_idx_2 = col_idxs[col_key_2]
                        if col_mtxs[col_idx_0][col_idx_2] is None:
                            col_mtxs[col_idx_0][col_idx_2] = col_mtx_02
                        else:                        
                            col_mtxs[col_idx_0][col_idx_2] += col_mtx_02
                
        # Global boundary convection matrix is not block-diagonal
        # but we arranged the column matrices in the proper form
        bdry_conv_mtx = sp.bmat(col_mtxs, format = 'csr')
        
    else:
        bdry_conv_mtx = 0
    
    if kwargs['verbose']:
        tf = perf_counter()
        msg = (
            'Constructed Boundary Propagation Matrix\n' +
            12 * ' '  + 'Time Elapsed: {:8.4f} [s]\n'.format(tf - t0)
        )
        utils.print_msg(msg)
    
    if kwargs['blocking']:        
        MPI_comm.Barrier()
        
    return bdry_conv_mtx

def calc_bdry_conv_matrix_mpi(mesh, **kwargs):
    
    default_kwargs = {'verbose'  : False, # Print info while executing
                      'blocking' : True   # Synchronize ranks before exiting
                      }
    kwargs = {**default_kwargs, **kwargs}
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    comm      = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()

    if comm_rank == 0:
        n_global = mesh.get_ndof()
    else:
        n_global = None
    n_global = MPI_comm.bcast(n_global, root = 0)
    
    if kwargs['verbose']:
        t0 = perf_counter()
        msg = (
            'Constructing Boundary Propagation Matrix...\n'
            )
        utils.print_msg(msg)
        
    # Calculate these matrices in serial, and then we'll split them
    if comm_rank == 0:
        # Variables that are the same throughout the loops
        col_items = sorted(mesh.cols.items())
        
        # Create column indexing for constructing global mass matrix
        [ncols, col_idxs] = proj.get_col_idxs(mesh)
        col_mtxs = [[None] * ncols for C in range(0, ncols)] # We have to assemble a
        # lot on inter-column interaction matrices,
        # so the construction is a bit more difficult.
        
        # The local-column matrices come in two kinds: M^CC and M^CC'.
        # The M^CC have to be constructed in four parts: M^CC_F.
        # The M^CC' can be constructed in one part.
        # We loop through each column C, then through each face F of C.
        # For each face, loop through each element K of C.
        # Depending on K, we contribute to M^CC_F or M^CC'.
        # Hold all four M^CC_F, add them together after all of the loops.
        for col_key_0, col_0 in col_items:
            if col_0.is_lf:
                # Use _0 to refer to column C
                # Later, use _1 to refer to column C'
                col_idx_0 = col_idxs[col_key_0]
                
                # Loop through the faces of C
                for F in range(0, 4):
                    [col_mtx_00, [col_key_1, col_mtx_01], [col_key_2, col_mtx_02]] = \
                        calc_col_matrix(mesh, col_key_0, F)
                    
                    # The intra-column matrix may already exist. If so, add to it.
                    if col_mtxs[col_idx_0][col_idx_0] is None:
                        col_mtxs[col_idx_0][col_idx_0] = col_mtx_00
                    else:                        
                        col_mtxs[col_idx_0][col_idx_0] += col_mtx_00
                        
                    if col_key_1 is not None:
                        col_idx_1 = col_idxs[col_key_1]
                        if col_mtxs[col_idx_0][col_idx_1] is None:
                            col_mtxs[col_idx_0][col_idx_1] = col_mtx_01
                        else:                        
                            col_mtxs[col_idx_0][col_idx_1] += col_mtx_01
                    if col_key_2 is not None:
                        col_idx_2 = col_idxs[col_key_2]
                        if col_mtxs[col_idx_0][col_idx_2] is None:
                            col_mtxs[col_idx_0][col_idx_2] = col_mtx_02
                        else:                        
                            col_mtxs[col_idx_0][col_idx_2] += col_mtx_02
                
        # Global boundary convection matrix is not block-diagonal
        # but we arranged the column matrices in the proper form
        bdry_conv_mtx = sp.bmat(col_mtxs, format = 'csr')
        
    else:
        bdry_conv_mtx = None
    
    if kwargs['verbose']:
        tf = perf_counter()
        msg = (
            'Constructed Boundary Propagation Matrix\n' +
            12 * ' '  + 'Time Elapsed: {:8.4f} [s]\n'.format(tf - t0)
        )
        utils.print_msg(msg)
        
    if kwargs['verbose']:
        t0 = perf_counter()
        msg = (
            'Scattering Boundary Propagation Matrix...\n'
            )
        utils.print_msg(msg)
        
    # Create PETSc sparse matrix
    M_MPI = PETSc.Mat()
    M_MPI.createAIJ(size = [n_global, n_global], comm = comm)
    
    o_rngs = M_MPI.getOwnershipRanges()
    ii_0 = o_rngs[comm_rank]
    ii_f = o_rngs[comm_rank+1]
    if comm_rank == 0:
        # Communicate global information
        for rank in range(1, comm_size):
            ii_0_else = o_rngs[rank]
            ii_f_else = o_rngs[rank+1]
            MPI_comm.send(bdry_conv_mtx[ii_0_else:ii_f_else, :],
                          dest = rank)
        M_local = bdry_conv_mtx[ii_0:ii_f, :]
        
    else:
        M_local = MPI_comm.recv(source = 0)
        
    # Put A_local into the shared matrix
    (II, JJ, VV) = sp.find(M_local)
    nnz_local    = np.size(II)
    for idx in range(0, nnz_local):
        ii = II[idx]
        jj = JJ[idx]
        vv = VV[idx]
        
        M_MPI[ii + ii_0, jj] = vv
        
    # Communicate off-rank values and setup internal data structures for
    # performing parallel operations
    M_MPI.assemblyBegin()
    M_MPI.assemblyEnd()
    
    if kwargs['blocking']:        
        MPI_comm.Barrier()
        
    return M_MPI

def calc_col_matrix(mesh, col_key_0, F):
    """
    Create the column matrices corresponding to face F of column 0.
    """

    tol = 1.e-15 # Tolerance for non-zero values in matrices.

    col_0 = mesh.cols[col_key_0]
    cell_items_0             = sorted(col_0.cells.items())
    [x0_0, y0_0, x1_0, y1_0] = col_0.pos[:]
    [dx_0, dy_0]             = [x1_0 - x0_0, y1_0 - y0_0]
    [nx_0, ny_0]             = col_0.ndofs
    [_, wx_0, _, wy_0, _, _] = qd.quad_xyth(nnodes_x = nx_0, nnodes_y = ny_0)
    [nc_0, cell_idxs_0]      = proj.get_cell_idxs(mesh, col_key_0)
    cell_mtxs_00             = [None] * nc_0

    # Get neighboring columns along face F
    [col_key_1, col_key_2] = col_0.nhbr_keys[F][:]
    if col_key_1 is not None:
        col_1 = mesh.cols[col_key_1]
        cell_items_1             = sorted(col_1.cells.items())
        [x0_1, y0_1, x1_1, y1_1] = col_1.pos[:]
        [dx_1, dy_1]             = [x1_1 - x0_1, y1_1 - y0_1]
        [nx_1, ny_1]             = col_1.ndofs[:]
        [nc_1, cell_idxs_1]      = proj.get_cell_idxs(mesh, col_key_1)
        cell_mtxs_01             = [[None] * nc_1 for K in range(0, nc_0)]
        if (F%2 == 0): # Construct E^K'K,y_jq
            E_x_01 = None
            E_y_01 = get_Ey(mesh, col_key_0, col_key_1)
        else: # F%2 == 1, construct E^K'K,x_ip
            E_x_01 = get_Ex(mesh, col_key_0, col_key_1)
            E_y_01 = None
    else:
        col_1               = None
        cell_items_1        = None
        [nx_1, ny_1]        = [None, None]
        [nc_1, cell_idxs_1] = [None, None]
        cell_mtxs_01        = None
        E_x_01 = None
        E_y_01 = None

    if (col_key_2 != col_key_1) and (col_key_2 is not None):
        col_2 = mesh.cols[col_key_2]
        cell_items_2             = sorted(col_2.cells.items())
        [x0_2, y0_2, x1_2, y1_2] = col_2.pos[:]
        [dx_2, dy_2]             = [x1_2 - x0_2, y1_2 - y0_2]
        [nx_2, ny_2]             = col_2.ndofs[:]
        [nc_2, cell_idxs_2]      = proj.get_cell_idxs(mesh, col_key_2)
        cell_mtxs_02             = [[None] * nc_2 for K in range(0, nc_0)]
        if (F%2 == 0): # Construct E^K'K,y_jq
            E_x_02 = None
            E_y_02 = get_Ey(mesh, col_key_0, col_key_2)
        else: # F%2 == 1, construct E^K'K,x_ip
            E_x_02 = get_Ex(mesh, col_key_0, col_key_2)
            E_y_02 = None
    else:
        col_key_2           = None
        col_2               = None
        cell_items_2        = None
        [nx_2, ny_2]        = [None, None]
        [nc_2, cell_idxs_2] = [None, None]
        cell_mtxs_02        = None
        E_x_02 = None
        E_y_02 = None
        
    # To ensure proper matrix construction, we initialize all cell
    # matrices to be empty sparse matrices
    for cell_key_0, cell_0 in cell_items_0:
        if cell_0.is_lf:
            cell_idx_0  = cell_idxs_0[cell_key_0]
            [nth_0]     = cell_0.ndofs[:]
            cell_ndof_0 = nx_0 * ny_0 * nth_0
            
            cell_mtxs_00[cell_idx_0] = \
                sp.coo_matrix((cell_ndof_0, cell_ndof_0))
            
            if cell_items_1 is not None:
                for cell_key_1, cell_1 in cell_items_1:
                    if cell_1.is_lf:
                        cell_idx_1  = cell_idxs_1[cell_key_1]
                        [nth_1]     = cell_1.ndofs[:]
                        cell_ndof_1 = nx_1 * ny_1 * nth_1
                        
                        cell_mtxs_01[cell_idx_0][cell_idx_1] = \
                            sp.coo_matrix((cell_ndof_0, cell_ndof_1))
                        
            if cell_items_2 is not None:
                for cell_key_2, cell_2 in cell_items_2:
                    if cell_2.is_lf:
                        cell_idx_2  = cell_idxs_2[cell_key_2]
                        [nth_2]     = cell_2.ndofs[:]
                        cell_ndof_2 = nx_2 * ny_2 * nth_2
                        
                        cell_mtxs_02[cell_idx_0][cell_idx_2] = \
                            sp.coo_matrix((cell_ndof_0, cell_ndof_2))
                    
    # Loop through cells of column C
    # For each cell in column C, we loop through the neighboring cells K^(n)
    # in neighboring column C'
    for cell_key_0, cell_0 in cell_items_0:
        if cell_0.is_lf:
            # Get information about cell K in column C
            [th0_0, th1_0] = cell_0.pos[:]
            dth_0 = th1_0 - th0_0
            [nth_0] = cell_0.ndofs[:]
            cell_ndof_0 = nx_0 * ny_0 * nth_0
            S_quad_0   = cell_0.quad
            cell_idx_0 = cell_idxs_0[cell_key_0] # Matrix index of cell 0 in
                                                 # column matrices
                                                 
            # If a cell is in F^+, contribute to column-matrix M^CC.
            is_Fp = ( ((S_quad_0 == 0) and (F == 0 or F == 1)) or
                      ((S_quad_0 == 1) and (F == 1 or F == 2)) or
                      ((S_quad_0 == 2) and (F == 2 or F == 3)) or
                      ((S_quad_0 == 3) and (F == 3 or F == 0)) )
            
            # Calculate values common across all cell matrices
            if (F%2 == 0):
                dcoeff = dy_0 * dth_0 / 4.
            else: # F%2 == 1
                dcoeff = dx_0 * dth_0 / 4.
                
            # If we're in Fp we contribute to M^CC and use the first formula
            # Otherwise we have the option of using the quadrature rule from
            # the neighboring column/cell
            if is_Fp:
                [_, _, _, _, thb_0, wth_0] = qd.quad_xyth(nnodes_th = nth_0)
                thf = proj.push_forward(th0_0, th1_0, thb_0)
                Th_F = Theta_F(thf, F)
                
                alpha = mat.get_idx_map(nx_0, ny_0, nth_0)
                beta  = mat.get_idx_map(nx_0, ny_0, nth_0)
                
                if (F%2 == 0):
                    alphalist = np.zeros([ny_0 * nth_0], dtype = np.int32)
                    betalist  = np.zeros([ny_0 * nth_0], dtype = np.int32)
                    vlist     = np.zeros([ny_0 * nth_0])
                    
                    if (F == 0):
                        x_idx = nx_0 - 1
                    else: # F == 2
                        x_idx = 0
                    
                    idx = 0
                    for jj in range(0, ny_0):
                        wy_j = wy_0[jj]
                        for aa in range(0, nth_0):
                            wth_a = wth_0[aa]
                            Th_F_a = Th_F[aa]
                            
                            val = dcoeff * wy_j * wth_a * Th_F_a
                            if np.abs(val) > tol:
                                alphalist[idx] = alpha(x_idx, jj, aa)
                                betalist[idx]  = beta(x_idx, jj, aa)
                                vlist[idx]     = val
                                
                                idx += 1
                            
                else: #F%2 == 1
                    alphalist = np.zeros([nx_0 * nth_0], dtype = np.int32)
                    betalist  = np.zeros([nx_0 * nth_0], dtype = np.int32)
                    vlist     = np.zeros([nx_0 * nth_0])
                    
                    if (F == 1):
                        y_idx = ny_0 - 1
                    else: # F == 3
                        y_idx = 0
                    
                    idx = 0
                    for ii in range(0, nx_0):
                        wx_i = wx_0[ii]
                        for aa in range(0, nth_0):
                            wth_a = wth_0[aa]
                            Th_F_a = Th_F[aa]
                            
                            val = dcoeff * wx_i * wth_a * Th_F_a
                            if np.abs(val) > tol:
                                alphalist[idx] = alpha(ii, y_idx, aa)
                                betalist[idx]  = beta(ii, y_idx, aa)
                                vlist[idx]     = val
                                
                                idx += 1
                
                cell_mtxs_00[cell_idx_0] = \
                    sp.coo_matrix((vlist, (alphalist, betalist)),
                               shape = (cell_ndof_0, cell_ndof_0))
                
                cell_mtxs_00[cell_idx_0].eliminate_zeros()
                
            else: # no is_Fp
                if col_key_1 is not None:
                    nhbr_cell_keys = ji_mesh.get_cell_nhbr_in_col(mesh,
                                                                  col_key_0,
                                                                  cell_key_0,
                                                                  col_key_1)
                    for cell_key_1 in nhbr_cell_keys:
                        if cell_key_1 is not None:
                            cell_1 = col_1.cells[cell_key_1]
                            if cell_1.is_lf:
                                cell_idx_1  = cell_idxs_1[cell_key_1]
                                [nth_1]     = cell_1.ndofs[:]
                                cell_ndof_1 = nx_1 * ny_1 * nth_1
                                
                                alpha = mat.get_idx_map(nx_0, ny_0, nth_0)
                                beta  = mat.get_idx_map(nx_1, ny_1, nth_1)
                                
                                E_th = get_Eth(mesh,
                                               col_key_0, cell_key_0,
                                               col_key_1, cell_key_1,
                                               F)
                                
                                if (F%2 == 0):
                                    max_len   = ny_0 * nth_0 * ny_1 * nth_1
                                    alphalist = np.zeros([max_len], dtype = np.int32)
                                    betalist  = np.zeros([max_len], dtype = np.int32)
                                    vlist     = np.zeros([max_len])
                                    
                                    if (F == 0):
                                        x_idx_0 = nx_0 - 1
                                        x_idx_1 = 0
                                    else: # F == 2
                                        x_idx_0 = 0
                                        x_idx_1 = nx_1 - 1
                                        
                                    idx = 0
                                    for jj in range(0, ny_1):
                                        for qq in range(0, ny_0):
                                            Ey_jq = E_y_01[jj, qq]
                                            for aa in range(0, nth_1):
                                                for rr in range(0, nth_0):
                                                    Eth_ar = E_th[aa, rr]
                                                    
                                                    val = dcoeff * Ey_jq * Eth_ar
                                                    
                                                    if np.abs(val) > tol:
                                                        alphalist[idx] = \
                                                            alpha(x_idx_0, qq, rr)
                                                        betalist[idx]  = \
                                                            beta( x_idx_1, jj, aa)
                                                        vlist[idx]     = val
                                                        idx += 1
                                                        
                                else: #F%2 == 1
                                    max_len   = nx_0 * nth_0 * nx_1 * nth_1
                                    alphalist = np.zeros([max_len], dtype = np.int32)
                                    betalist  = np.zeros([max_len], dtype = np.int32)
                                    vlist     = np.zeros([max_len])
                                    
                                    if (F == 1):
                                        y_idx_0 = ny_0 - 1
                                        y_idx_1 = 0
                                    else: # F == 3
                                        y_idx_0 = 0
                                        y_idx_1 = ny_1 - 1
                                        
                                    idx = 0
                                    for ii in range(0, nx_1):
                                        for pp in range(0, nx_0):
                                            Ex_ip = E_x_01[ii, pp]
                                            for aa in range(0, nth_1):
                                                for rr in range(0, nth_0):
                                                    Eth_ar = E_th[aa, rr]
                                                    
                                                    val = dcoeff * Ex_ip * Eth_ar
                                                    
                                                    if np.abs(val) > tol:
                                                        alphalist[idx] = \
                                                            alpha(pp, y_idx_0, rr)
                                                        betalist[idx]  = \
                                                            beta( ii, y_idx_1, aa)
                                                        vlist[idx]     = val
                                                        idx += 1
                                                        
                                cell_mtxs_01[cell_idx_0][cell_idx_1] = \
                                    sp.coo_matrix((vlist, (alphalist, betalist)),
                                               shape = (cell_ndof_0, cell_ndof_1))
                                
                                cell_mtxs_01[cell_idx_0][cell_idx_1].eliminate_zeros()
                        
                if col_key_2 is not None:
                    nhbr_cell_keys = ji_mesh.get_cell_nhbr_in_col(mesh,
                                                                  col_key_0,
                                                                  cell_key_0,
                                                                  col_key_2)
                    for cell_key_2 in nhbr_cell_keys:
                        if cell_key_2 is not None:
                            cell_2 = col_2.cells[cell_key_2]
                            if cell_2.is_lf:
                                cell_idx_2  = cell_idxs_2[cell_key_2]
                                [nth_2]     = cell_2.ndofs[:]
                                cell_ndof_2 = nx_2 * ny_2 * nth_2
                                
                                alpha = mat.get_idx_map(nx_0, ny_0, nth_0)
                                beta  = mat.get_idx_map(nx_2, ny_2, nth_2)
                                
                                E_th = get_Eth(mesh,
                                               col_key_0, cell_key_0,
                                               col_key_2, cell_key_2,
                                               F)
                                
                                if (F%2 == 0):
                                    max_len   = ny_0 * nth_0 * ny_2 * nth_2
                                    alphalist = np.zeros([max_len], dtype = np.int32)
                                    betalist  = np.zeros([max_len], dtype = np.int32)
                                    vlist     = np.zeros([max_len])
                                    
                                    if (F == 0):
                                        x_idx_0 = nx_0 - 1
                                        x_idx_2 = 0
                                    else: # F == 2
                                        x_idx_0 = 0
                                        x_idx_2 = nx_2 - 1
                                        
                                    idx = 0
                                    for jj in range(0, ny_2):
                                        for qq in range(0, ny_0):
                                            Ey_jq = E_y_02[jj, qq]
                                            for aa in range(0, nth_2):
                                                for rr in range(0, nth_0):
                                                    Eth_ar = E_th[aa, rr]
                                                    
                                                    val = dcoeff * Ey_jq * Eth_ar
                                                    
                                                    if np.abs(val) > tol:
                                                        alphalist[idx] = \
                                                            alpha(x_idx_0, qq, rr)
                                                        betalist[idx]  = \
                                                            beta( x_idx_2, jj, aa)
                                                        vlist[idx]     = val
                                                        idx += 1
                                                        
                                else: #F%2 == 1
                                    max_len   = nx_0 * nth_0 * nx_2 * nth_2
                                    alphalist = np.zeros([max_len], dtype = np.int32)
                                    betalist  = np.zeros([max_len], dtype = np.int32)
                                    vlist     = np.zeros([max_len])
                                    
                                    if (F == 1):
                                        y_idx_0 = ny_0 - 1
                                        y_idx_2 = 0
                                    else: # F == 3
                                        y_idx_0 = 0
                                        y_idx_2 = ny_2 - 1
                                        
                                    idx = 0
                                    for ii in range(0, nx_2):
                                        for pp in range(0, nx_0):
                                            Ex_ip = E_x_02[ii, pp]
                                            for aa in range(0, nth_2):
                                                for rr in range(0, nth_0):
                                                    Eth_ar = E_th[aa, rr]
                                                    
                                                    val = dcoeff * Ex_ip * Eth_ar
                                                    
                                                    if np.abs(val) > tol:
                                                        alphalist[idx] = \
                                                            alpha(pp, y_idx_0, rr)
                                                        betalist[idx]  = \
                                                            beta( ii, y_idx_2, aa)
                                                        vlist[idx]     = val
                                                        idx += 1
                                cell_mtxs_02[cell_idx_0][cell_idx_2] = \
                                    sp.coo_matrix((vlist, (alphalist, betalist)),
                                               shape = (cell_ndof_0, cell_ndof_2))
                        
                                cell_mtxs_02[cell_idx_0][cell_idx_2].eliminate_zeros()
                
    col_mtx_00 = sp.block_diag(cell_mtxs_00, format = 'coo')
    if col_1 is not None:
        col_mtx_01 = sp.bmat(cell_mtxs_01, format = 'coo')
    else:
        col_mtx_01 = None
        
    if col_2 is not None:
        col_mtx_02 = sp.bmat(cell_mtxs_02, format = 'coo')
    else:
        col_mtx_02 = None
        
    return [col_mtx_00, [col_key_1, col_mtx_01], [col_key_2, col_mtx_02]]

def Theta_F(theta, F):
    return np.cos(theta - F * np.pi / 2.)
