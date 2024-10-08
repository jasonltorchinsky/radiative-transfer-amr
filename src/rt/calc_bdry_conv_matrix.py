# Standard Library Imports
from time import perf_counter

# Third-Party Library Imports
import numpy        as np
import petsc4py
import scipy.sparse as sp
from   mpi4py       import MPI
from   petsc4py     import PETSc

# Local Library Imports
import consts
from dg.matrix import get_col_idxs, get_cell_idxs
from dg.mesh import Mesh
from dg.mesh.column import Column
from dg.mesh.column.cell import Cell
from dg.matrix import get_idx_map
from dg.projection import push_forward
from dg.quadrature import quad_xyth
import utils

# Relative Imports
from .get_Ex  import get_Ex
from .get_Ey  import get_Ey
from .get_Eth import get_Eth

def calc_bdry_conv_matrix(mesh: Mesh, **kwargs) -> sp._csr.csr_matrix: # Output typing might be wrong
    
    default_kwargs: dict = {"verbose"  : False, # Print info while executing
                      "blocking" : True   # Synchronize ranks before exiting
                      }
    kwargs: dict = {**default_kwargs, **kwargs}
    
    # Initialize parallel communicators
    MPI_comm: MPI.Intracomm = MPI.COMM_WORLD
    
    if not PETSc.Sys.isInitialized():
        petsc4py.init(comm = MPI_comm)
    PETSc_comm: PETSc.Comm = PETSc.COMM_WORLD
    comm_rank: int = PETSc_comm.getRank()
    comm_size: int = PETSc_comm.getSize()

    if comm_rank == 0:
        n_global: int = mesh.get_ndof()
    else:
        n_global: int = None
    n_global: int = MPI_comm.bcast(n_global, root = 0)
    
    if kwargs["verbose"]:
        t0: float = perf_counter()
        msg: str = ( "Constructing Boundary Propagation Matrix...\n" )
        utils.print_msg(msg)
        
    # Calculate these matrices in serial, and then we"ll split them
    if comm_rank == 0:
        # Variables that are the same throughout the loops
        col_items: list = sorted(mesh.cols.items())
        
        # Create column indexing for constructing global mass matrix
        [ncols, col_idxs] = get_col_idxs(mesh)
        col_mtxs: list = [[None] * ncols for C in range(0, ncols)] # We have to assemble a
        # lot on inter-column interaction matrices,
        # so the construction is a bit more difficult.
        
        # The local-column matrices come in two kinds: M^CC and M^CC".
        # The M^CC have to be constructed in four parts: M^CC_F.
        # The M^CC" can be constructed in one part.
        # We loop through each column C, then through each face F of C.
        # For each face, loop through each element K of C.
        # Depending on K, we contribute to M^CC_F or M^CC".
        # Hold all four M^CC_F, add them together after all of the loops.
        for col_key_0, col_0 in col_items:
            assert(col_0.is_lf)
                
            # Use _0 to refer to column C
            # Later, use _1 to refer to column C"
            col_idx_0: list = col_idxs[col_key_0]
            
            # Loop through the faces of C
            for col_face_0 in range(0, 4):
                [col_mtx_00, [col_key_1, col_mtx_01], [col_key_2, col_mtx_02]] = \
                    calc_col_matrix(mesh, col_key_0, col_face_0)
                
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
        bdry_conv_mtx: sp._csr.csr_matrix = sp.bmat(col_mtxs, format = "csr")
        
    else:
        bdry_conv_mtx: sp._csr.csr_matrix = None
    
    if kwargs["verbose"]:
        tf: float = perf_counter()
        msg: str = ( "Constructed Boundary Propagation Matrix\n" +
                     12 * " "  + "Time Elapsed: {:8.4f} [s]\n".format(tf - t0) )
        utils.print_msg(msg)
        
    if kwargs["verbose"]:
        t0: float = perf_counter()
        msg: str = ( "Scattering Boundary Propagation Matrix...\n" )
        utils.print_msg(msg)
        
    # Create PETSc sparse matrix
    M_MPI: PETSc.Mat = PETSc.Mat()
    M_MPI.createAIJ(size = [n_global, n_global], comm = PETSc_comm)
    
    o_rngs: list = M_MPI.getOwnershipRanges()
    ii_0: list = o_rngs[comm_rank]
    ii_f: list = o_rngs[comm_rank+1]
    if comm_rank == 0:
        # Communicate global information
        for rank in range(1, comm_size):
            ii_0_else: list = o_rngs[rank]
            ii_f_else: list = o_rngs[rank+1]
            MPI_comm.send(bdry_conv_mtx[ii_0_else:ii_f_else, :],
                          dest = rank)
        M_local: sp._csr.csr_matrix = bdry_conv_mtx[ii_0:ii_f, :]
        
    else:
        M_local: sp._csr.csr_matrix = MPI_comm.recv(source = 0)
        
    # Put A_local into the shared matrix
    (II, JJ, VV) = sp.find(M_local)
    nnz_local: int = np.size(II)
    for idx in range(0, nnz_local):
        ii: int = II[idx]
        jj: int = JJ[idx]
        vv: float = VV[idx]
        
        M_MPI[ii + ii_0, jj] = vv
        
    # Communicate off-rank values and setup internal data structures for
    # performing parallel operations
    M_MPI.assemblyBegin()
    M_MPI.assemblyEnd()
    
    if kwargs["blocking"]:        
        MPI_comm.Barrier()
        
    return M_MPI

def calc_col_matrix(mesh: Mesh, col_key_0: int, col_face_0: int) -> list:
    """
    Create the column matrices corresponding to face F of column 0.
    """

    tol: float = 1.e-15 # Tolerance for non-zero values in matrices.

    col_0: Column = mesh.cols[col_key_0]
    cell_items_0: list = sorted(col_0.cells.items())
    [x0_0, y0_0, x1_0, y1_0] = col_0.pos[:]
    [dx_0, dy_0]             = [x1_0 - x0_0, y1_0 - y0_0]
    [nx_0, ny_0]             = col_0.ndofs
    [_, wx_0, _, wy_0, _, _] = quad_xyth(nnodes_x = nx_0, nnodes_y = ny_0)
    [nc_0, cell_idxs_0]      = get_cell_idxs(mesh, col_key_0)
    cell_mtxs_00: list       = [None] * nc_0

    # Get neighboring columns along face F
    [col_key_1, col_key_2] = col_0.nhbr_keys[col_face_0][:]
    if col_key_1 is not None:
        col_1: Column = mesh.cols[col_key_1]
        cell_items_1: list = sorted(col_1.cells.items())
        [nx_1, ny_1] = col_1.ndofs[:]
        [nc_1, cell_idxs_1] = get_cell_idxs(mesh, col_key_1)
        cell_mtxs_01: list  = [[None] * nc_1 for K in range(0, nc_0)]
        if (col_face_0%2 == 0): # Construct E^K"K,y_jq
            E_x_01: np.ndarray = None
            E_y_01: np.ndarray = get_Ey(mesh, col_key_0, col_key_1)
        else: # F%2 == 1, construct E^K"K,x_ip
            E_x_01: np.ndarray = get_Ex(mesh, col_key_0, col_key_1)
            E_y_01: np.ndarray = None
    else:
        col_1: Column = None
        cell_items_1: list = None
        [nx_1, ny_1]        = [None, None]
        [nc_1, cell_idxs_1] = [None, None]
        cell_mtxs_01: list = None
        E_x_01: np.ndarray = None
        E_y_01: np.ndarray = None

    if (col_key_2 != col_key_1) and (col_key_2 is not None):
        col_2: Column = mesh.cols[col_key_2]
        cell_items_2: list = sorted(col_2.cells.items())
        [nx_2, ny_2] = col_2.ndofs[:]
        [nc_2, cell_idxs_2] = get_cell_idxs(mesh, col_key_2)
        cell_mtxs_02: list = [[None] * nc_2 for K in range(0, nc_0)]
        if (col_face_0%2 == 0): # Construct E^K"K,y_jq
            E_x_02: np.ndarray = None
            E_y_02: np.ndarray = get_Ey(mesh, col_key_0, col_key_2)
        else: # F%2 == 1, construct E^K"K,x_ip
            E_x_02: np.ndarray = get_Ex(mesh, col_key_0, col_key_2)
            E_y_02: np.ndarray = None
    else:
        col_key_2: int = None
        col_2: Column  = None
        cell_items_2: list = None
        [nx_2, ny_2] = [None, None]
        [nc_2, cell_idxs_2] = [None, None]
        cell_mtxs_02: list = None
        E_x_02: np.ndarray = None
        E_y_02: np.ndarray = None
        
    # To ensure proper matrix construction, we initialize all cell
    # matrices to be empty sparse matrices
    for cell_key_0, cell_0 in cell_items_0:
        assert(cell_0.is_lf)
        
        cell_idx_0: int = cell_idxs_0[cell_key_0]
        [nth_0] = cell_0.ndofs[:]
        cell_ndof_0: int = nx_0 * ny_0 * nth_0
        
        cell_mtxs_00[cell_idx_0] = \
            sp.coo_matrix((cell_ndof_0, cell_ndof_0))
        
        if cell_items_1 is not None:
            for cell_key_1, cell_1 in cell_items_1:
                assert(cell_1.is_lf)
                    
                cell_idx_1: int = cell_idxs_1[cell_key_1]
                [nth_1] = cell_1.ndofs[:]
                cell_ndof_1: int = nx_1 * ny_1 * nth_1
                
                cell_mtxs_01[cell_idx_0][cell_idx_1] = \
                    sp.coo_matrix((cell_ndof_0, cell_ndof_1))
                    
        if cell_items_2 is not None:
            for cell_key_2, cell_2 in cell_items_2:
                assert(cell_2.is_lf)

                cell_idx_2: int  = cell_idxs_2[cell_key_2]
                [nth_2] = cell_2.ndofs[:]
                cell_ndof_2: int = nx_2 * ny_2 * nth_2
                
                cell_mtxs_02[cell_idx_0][cell_idx_2] = \
                    sp.coo_matrix((cell_ndof_0, cell_ndof_2))
                    
    # Loop through cells of column C
    # For each cell in column C, we loop through the neighboring cells K^(n)
    # in neighboring column C"
    for cell_key_0, cell_0 in cell_items_0:
        assert(cell_0.is_lf)
        # Get information about cell K in column C
        [th0_0, th1_0] = cell_0.pos[:]
        dth_0: float = th1_0 - th0_0
        [nth_0] = cell_0.ndofs[:]
        cell_ndof_0: int = nx_0 * ny_0 * nth_0
        S_quad_0: int   = cell_0.quadrant
        cell_idx_0: int = cell_idxs_0[cell_key_0] # Matrix index of cell 0 in
                                             # column matrices
                                             
        # If a cell is in F^+, contribute to column-matrix M^CC.
        is_Fp: bool = ( ((S_quad_0 == 0) and (col_face_0 == 0 or col_face_0 == 1)) or
                        ((S_quad_0 == 1) and (col_face_0 == 1 or col_face_0 == 2)) or
                        ((S_quad_0 == 2) and (col_face_0 == 2 or col_face_0 == 3)) or
                        ((S_quad_0 == 3) and (col_face_0 == 3 or col_face_0 == 0)) )
        
        # Calculate values common across all cell matrices
        if (col_face_0%2 == 0):
            dcoeff: float = dy_0 * dth_0 / 4.
        else: # F%2 == 1
            dcoeff: float = dx_0 * dth_0 / 4.
            
        # If we"re in Fp we contribute to M^CC and use the first formula
        # Otherwise we have the option of using the quadrature rule from
        # the neighboring column/cell
        if is_Fp:
            [_, _, _, _, thb_0, wth_0] = quad_xyth(nnodes_th = nth_0)
            thf: np.ndarray = push_forward(th0_0, th1_0, thb_0)
            Th_F: np.ndarray = Theta_F(thf, col_face_0)
            
            alpha = get_idx_map(nx_0, ny_0, nth_0)
            beta  = get_idx_map(nx_0, ny_0, nth_0)
            
            if (col_face_0%2 == 0):
                alphalist: np.ndarray = np.zeros([ny_0 * nth_0], dtype = consts.INT)
                betalist: np.ndarray  = np.zeros([ny_0 * nth_0], dtype = consts.INT)
                vlist: np.ndarray     = np.zeros([ny_0 * nth_0])
                
                if (col_face_0 == 0):
                    x_idx: int = nx_0 - 1
                else: # F == 2
                    x_idx: int = 0
                
                idx: int = 0
                for jj in range(0, ny_0):
                    wy_j: float = wy_0[jj]
                    for aa in range(0, nth_0):
                        wth_a: float = wth_0[aa]
                        Th_F_a: float = Th_F[aa]
                        
                        val: float = dcoeff * wy_j * wth_a * Th_F_a
                        if np.abs(val) > tol:
                            alphalist[idx] = alpha(x_idx, jj, aa)
                            betalist[idx]  = beta(x_idx, jj, aa)
                            vlist[idx]     = val
                            
                            idx += 1
                        
            else: #F%2 == 1
                alphalist: np.ndarray = np.zeros([nx_0 * nth_0], dtype = consts.INT)
                betalist: np.ndarray  = np.zeros([nx_0 * nth_0], dtype = consts.INT)
                vlist: np.ndarray     = np.zeros([nx_0 * nth_0])
                
                if (col_face_0 == 1):
                    y_idx: int = ny_0 - 1
                else: # F == 3
                    y_idx: int = 0
                
                idx: int = 0
                for ii in range(0, nx_0):
                    wx_i: float = wx_0[ii]
                    for aa in range(0, nth_0):
                        wth_a: float = wth_0[aa]
                        Th_F_a: float = Th_F[aa]
                        
                        val: float = dcoeff * wx_i * wth_a * Th_F_a
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
                nhbr_cell_keys: list = mesh.nhbr_cells_in_nhbr_col(col_key_0,
                                                                   cell_key_0,
                                                                   col_key_1)
                for cell_key_1 in nhbr_cell_keys:
                    if cell_key_1 is not None:
                        cell_1: Cell = col_1.cells[cell_key_1]
                        assert(cell_1.is_lf)
                            
                        cell_idx_1: int  = cell_idxs_1[cell_key_1]
                        [nth_1]     = cell_1.ndofs[:]
                        cell_ndof_1: int = nx_1 * ny_1 * nth_1
                        
                        alpha = get_idx_map(nx_0, ny_0, nth_0)
                        beta  = get_idx_map(nx_1, ny_1, nth_1)
                        
                        E_th: np.ndarray = get_Eth(mesh,
                                                   col_key_0, cell_key_0,
                                                   col_key_1, cell_key_1,
                                                   col_face_0)
                        
                        if (col_face_0%2 == 0):
                            max_len: int = ny_0 * nth_0 * ny_1 * nth_1
                            alphalist: np.ndarray = np.zeros([max_len], dtype = consts.INT)
                            betalist: np.ndarray  = np.zeros([max_len], dtype = consts.INT)
                            vlist: np.ndarray     = np.zeros([max_len])
                            
                            if (col_face_0 == 0):
                                x_idx_0: int = nx_0 - 1
                                x_idx_1: int = 0
                            else: # F == 2
                                x_idx_0: int = 0
                                x_idx_1: int = nx_1 - 1
                                
                            idx: int = 0
                            for jj in range(0, ny_1):
                                for qq in range(0, ny_0):
                                    Ey_jq: float = E_y_01[jj, qq]
                                    for aa in range(0, nth_1):
                                        for rr in range(0, nth_0):
                                            Eth_ar: float = E_th[aa, rr]
                                            
                                            val: float = dcoeff * Ey_jq * Eth_ar
                                            
                                            if np.abs(val) > tol:
                                                alphalist[idx] = \
                                                    alpha(x_idx_0, qq, rr)
                                                betalist[idx] = \
                                                    beta( x_idx_1, jj, aa)
                                                vlist[idx] = val
                                                idx += 1
                                                
                        else: #F%2 == 1
                            max_len: int = nx_0 * nth_0 * nx_1 * nth_1
                            alphalist: np.ndarray = np.zeros([max_len], dtype = consts.INT)
                            betalist: np.ndarray  = np.zeros([max_len], dtype = consts.INT)
                            vlist: np.ndarray     = np.zeros([max_len])
                            
                            if (col_face_0 == 1):
                                y_idx_0: int = ny_0 - 1
                                y_idx_1: int = 0
                            else: # F == 3
                                y_idx_0: int = 0
                                y_idx_1: int = ny_1 - 1
                                
                            idx: int = 0
                            for ii in range(0, nx_1):
                                for pp in range(0, nx_0):
                                    Ex_ip: float = E_x_01[ii, pp]
                                    for aa in range(0, nth_1):
                                        for rr in range(0, nth_0):
                                            Eth_ar: float = E_th[aa, rr]
                                            
                                            val: float = dcoeff * Ex_ip * Eth_ar
                                            
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
                nhbr_cell_keys: list = mesh.nhbr_cells_in_nhbr_col(col_key_0,
                                                                   cell_key_0,
                                                                   col_key_2)
                for cell_key_2 in nhbr_cell_keys:
                    if cell_key_2 is not None:
                        cell_2: Cell = col_2.cells[cell_key_2]
                        assert(cell_2.is_lf)
                        
                        cell_idx_2: int = cell_idxs_2[cell_key_2]
                        [nth_2] = cell_2.ndofs[:]
                        cell_ndof_2: int = nx_2 * ny_2 * nth_2
                        
                        alpha = get_idx_map(nx_0, ny_0, nth_0)
                        beta  = get_idx_map(nx_2, ny_2, nth_2)
                        
                        E_th: np.ndarray = get_Eth(mesh,
                                                   col_key_0, cell_key_0,
                                                   col_key_2, cell_key_2,
                                                   col_face_0)
                        
                        if (col_face_0%2 == 0):
                            max_len: int = ny_0 * nth_0 * ny_2 * nth_2
                            alphalist: np.ndarray = np.zeros([max_len], dtype = consts.INT)
                            betalist: np.ndarray  = np.zeros([max_len], dtype = consts.INT)
                            vlist: np.ndarray     = np.zeros([max_len])
                            
                            if (col_face_0 == 0):
                                x_idx_0: int = nx_0 - 1
                                x_idx_2: int = 0
                            else: # F == 2
                                x_idx_0: int = 0
                                x_idx_2: int = nx_2 - 1
                                
                            idx: int = 0
                            for jj in range(0, ny_2):
                                for qq in range(0, ny_0):
                                    Ey_jq: float = E_y_02[jj, qq]
                                    for aa in range(0, nth_2):
                                        for rr in range(0, nth_0):
                                            Eth_ar: float = E_th[aa, rr]
                                            
                                            val: float = dcoeff * Ey_jq * Eth_ar
                                            
                                            if np.abs(val) > tol:
                                                alphalist[idx] = \
                                                    alpha(x_idx_0, qq, rr)
                                                betalist[idx] = \
                                                    beta( x_idx_2, jj, aa)
                                                vlist[idx] = val
                                                idx += 1
                                                
                        else: #F%2 == 1
                            max_len: int = nx_0 * nth_0 * nx_2 * nth_2
                            alphalist: np.ndarray = np.zeros([max_len], dtype = consts.INT)
                            betalist: np.ndarray  = np.zeros([max_len], dtype = consts.INT)
                            vlist: np.ndarray     = np.zeros([max_len])
                            
                            if (col_face_0 == 1):
                                y_idx_0: int = ny_0 - 1
                                y_idx_2: int = 0
                            else: # F == 3
                                y_idx_0: int = 0
                                y_idx_2: int = ny_2 - 1
                                
                            idx: int = 0
                            for ii in range(0, nx_2):
                                for pp in range(0, nx_0):
                                    Ex_ip: float = E_x_02[ii, pp]
                                    for aa in range(0, nth_2):
                                        for rr in range(0, nth_0):
                                            Eth_ar: float = E_th[aa, rr]
                                            
                                            val: float = dcoeff * Ex_ip * Eth_ar
                                            
                                            if np.abs(val) > tol:
                                                alphalist[idx] = \
                                                    alpha(pp, y_idx_0, rr)
                                                betalist[idx] = \
                                                    beta( ii, y_idx_2, aa)
                                                vlist[idx] = val
                                                idx += 1
                        cell_mtxs_02[cell_idx_0][cell_idx_2] = \
                            sp.coo_matrix((vlist, (alphalist, betalist)),
                                       shape = (cell_ndof_0, cell_ndof_2))
                
                        cell_mtxs_02[cell_idx_0][cell_idx_2].eliminate_zeros()
            
    col_mtx_00: sp._coo.coo_matrix = sp.block_diag(cell_mtxs_00, format = "coo")
    if col_1 is not None:
        col_mtx_01: sp._coo.coo_matrix = sp.bmat(cell_mtxs_01, format = "coo")
    else:
        col_mtx_01: sp._coo.coo_matrix = None
        
    if col_2 is not None:
        col_mtx_02: sp._coo.coo_matrix = sp.bmat(cell_mtxs_02, format = "coo")
    else:
        col_mtx_02: sp._coo.coo_matrix = None
        
    return [col_mtx_00, [col_key_1, col_mtx_01], [col_key_2, col_mtx_02]]

def Theta_F(theta: np.ndarray, col_face: int) -> np.ndarray:
    return np.cos(theta - col_face * consts.PI / 2.)
