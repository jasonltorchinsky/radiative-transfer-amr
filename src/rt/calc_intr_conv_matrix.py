# Standard Library Imports
from time import perf_counter

# Third-Party Library Imports
import numpy as np
import petsc4py
import scipy.sparse as sp
from   mpi4py   import MPI
from   petsc4py import PETSc

# Local Library Imports
import consts
from dg.matrix import get_idx_map, get_cell_idxs
from dg.mesh import Mesh
from dg.mesh.column import Column
from dg.projection import push_forward
from dg.quadrature import lag_ddx_eval, quad_xyth
import utils

# Relative Imports

ddy_psis: dict = {}
ddx_phis: dict = {}

def calc_intr_conv_matrix(mesh: Mesh, **kwargs) -> sp._coo.coo_matrix:
    default_kwargs: Mesh = {"verbose"      : False, # Print info while executing
                      "blocking"     : True   # Synchronize ranks before exiting
                      }
    kwargs: Mesh = {**default_kwargs, **kwargs}

    # Initialize parallel communicators
    MPI_comm: MPI.Intracomm = MPI.COMM_WORLD
    
    if not PETSc.Sys.isInitialized():
        petsc4py.init(comm = MPI_comm)
    PETSc_comm: PETSc.Comm = PETSc.COMM_WORLD
    comm_rank: int = PETSc_comm.getRank()
    comm_size: int = PETSc_comm.getSize()
    
    if kwargs["verbose"]:
        t0: float = perf_counter()
        msg: str = ( "Constructing Interior Propagation Matrix...\n" )
        utils.print_msg(msg)
    
    if comm_rank == 0:
        n_global: int = mesh.get_ndof()
    else:
        n_global: int = None
    n_global: int = MPI_comm.bcast(n_global, root = 0)
    
    # Split the problem into parts dependent on size of COMM_WORLD.
    col_keys_global: list = list(sorted(mesh.cols.keys()))
    col_keys_local: np.ndarray = np.array_split(col_keys_global, comm_size)[comm_rank].astype(consts.INT)
    
    # Get the start indices for each column matrix
    col_st_idxs: dict = {col_keys_global[0] : 0}
    col_ndofs: dict   = {}
    
    dof_count: int = 0
    for cc in range(1, len(col_keys_global)):
        col_key: int = col_keys_global[cc]
        prev_col_key: int = col_keys_global[cc - 1]
        prev_col: Column = mesh.cols[prev_col_key]
        
        assert(prev_col.is_lf)
        
        [ndof_x, ndof_y] = prev_col.ndofs[:]
        
        cell_items: list = sorted(prev_col.cells.items())
        for _, cell in cell_items:
            assert(cell.is_lf)
            
            [ndof_th]  = cell.ndofs[:]
            dof_count += ndof_x * ndof_y * ndof_th
        col_st_idxs[col_key] = dof_count
    col_ndofs[col_keys_global[-1]] = n_global - dof_count
    
    # Create PETSc sparse matrix
    M_MPI: PETSc.Mat = PETSc.Mat()
    M_MPI.createAIJ(size = [n_global, n_global], comm = PETSc_comm)

    # We assemble the column matrices using the block construction of scipy
    for col_key in col_keys_local:
        col_mtx: sp._csr.csr_matrix = calc_col_matrix(mesh, col_key)
        col_st_idx: list = col_st_idxs[col_key]
        (II, JJ, VV) = sp.find(col_mtx)
        nnz_local: int = np.size(II)
        for idx in range(0, nnz_local):
            ii: int = II[idx]
            jj: int = JJ[idx]
            vv: float = VV[idx]
            M_MPI[col_st_idx + ii, col_st_idx + jj] = vv
            
    # Communicate off-rank values and setup internal data structures for
    # performing parallel operations
    M_MPI.assemblyBegin()
    M_MPI.assemblyEnd()
    
    if kwargs["verbose"]:
        tf: float = perf_counter()
        msg: str = ( "Constructed Interior Propagation Matrix\n" +
                     12 * " "  + "Time Elapsed: {:8.4f} [s]\n".format(tf - t0) )
        utils.print_msg(msg)
       
    if kwargs["blocking"]:
        MPI_comm.Barrier()
        
    return M_MPI

def calc_col_matrix(mesh: Mesh, col_key: int) -> sp._csr.csr_matrix:
    col: Column = mesh.cols[col_key]
    if col.is_lf:
        # Get column information, quadrature weights
        [x0, y0, x1, y1] = col.pos
        [dx, dy] = [x1 - x0, y1 - y0]
        [nx, ny] = col.ndofs
        
        [xxb, wx, yyb, wy, _, _] = quad_xyth(nnodes_x = nx, nnodes_y = ny)
        
        wx: np.ndarray = wx.reshape([nx, 1])
        wy: np.ndarray = wy.reshape([1, ny])
        
        wx_wy: np.ndarray = wx * wy
        
        # Create cell indexing for constructing column matrix
        [ncells, cell_idxs] = get_cell_idxs(mesh, col_key)
        cell_mtxs: list = [None] * ncells # Column interior convection  matrix is
        # block-diagonal, and so there are only 
        # ncell non-zero cell interior convection
        # matrices
        
        for cell_key, cell in sorted(col.cells.items()):
            assert(cell.is_lf)

            # Get cell information, quadrature weights
            cell_idx: int = cell_idxs[cell_key]
            [th0, th1] = cell.pos
            dth: float = th1 - th0
            [nth] = cell.ndofs
            
            [_, _, _, _, thb, wth] = quad_xyth(nnodes_th = nth)
            
            thf: np.ndarray = push_forward(th0, th1, thb)
            
            wth_sinth: np.ndarray = wth * np.sin(thf)
            wth_costh: np.ndarray = wth * np.cos(thf)
            
            # Indexing from i, j, a to beta
            # Same formula for p, q, r to alpha, but we define alpha
            # anyway for clarity
            alpha = get_idx_map(nx, ny, nth)
            beta  = get_idx_map(nx, ny, nth)
            
            # Set up arrays for delta_ip * delta_ar term
            cell_ndof: int = nx * ny * nth
            cell_ndof_ipar: int = nx * ny**2 * nth
            alphalist_ipar: np.ndarray = np.zeros([cell_ndof_ipar], dtype = consts.INT)
            betalist_ipar: np.ndarray  = np.zeros([cell_ndof_ipar], dtype = consts.INT)
            vlist_ipar: np.ndarray     = np.zeros([cell_ndof_ipar])
            
            # Construct delta_ip * delta_ar term
            # i = p, a = r
            # When we take the derivative of psi, we end up with
            # 2/dy * psi_bar in normalized coordinates, so dcoeff is
            dcoeff: float = dx * dth / 4.
            
            # Store the ddy_psi matrix if we haven"t yet calculated it
            global ddy_psis
            if ny in ddy_psis.keys():
                ddy_psi: np.ndarray = ddy_psis[ny][:,:]
            else:
                ddy_psi: np.ndarray = np.zeros([ny, ny])
                for qq in range(0, ny):
                    for jj in range(0, ny):
                        ddy_psi[qq, jj] = lag_ddx_eval(yyb, qq, yyb[jj])
                ddy_psis[ny] = ddy_psi[:,:]
            
            idx: int = 0
            for ii in range(0, nx):
                for jj in range(0, ny):
                    wxi_wyj = wx_wy[ii, jj]
                    for aa in range(0, nth):
                        wth_sinth_a: float = wth_sinth[aa]
                        for qq in range(0, ny):
                            ddy_psi_qj: float = ddy_psi[qq, jj]
                            
                            alphalist_ipar[idx] = alpha(ii, qq, aa)
                            betalist_ipar[idx]  = beta( ii, jj, aa)
                            
                            vlist_ipar[idx] = dcoeff * wxi_wyj \
                                * wth_sinth_a * ddy_psi_qj
                            
                            idx += 1
                            
            delta_ipar: sp._coo.coo_matrix = sp.coo_matrix((vlist_ipar,
                                        (alphalist_ipar, betalist_ipar)),
                                       shape = (cell_ndof, cell_ndof))
            
            # Set up arrays for  delta_jq * delta_ar term
            cell_ndof_jqar: int = nx**2 * ny * nth
            alphalist_jqar: np.ndarray = np.zeros([cell_ndof_jqar], dtype = consts.INT)
            betalist_jqar: np.ndarray  = np.zeros([cell_ndof_jqar], dtype = consts.INT)
            vlist_jqar: np.ndarray     = np.zeros([cell_ndof_jqar])
            
            # Construct delta_jq * delta_ar term
            # j = q, a = r
            # When we take the derivative of phi, we end up with
            # 2/dx * phi_bar in normalized coordinates, so dcoeff is
            dcoeff: float = dy * dth / 4.
            
            # Store the ddy_psi matrix if we haven"t yet calculated it
            global ddx_phis
            if nx in ddx_phis.keys():
                ddx_phi: np.ndarray = ddx_phis[nx][:,:]
            else:
                ddx_phi: np.ndarray = np.zeros([nx, nx])
                for pp in range(0, nx):
                    for ii in range(0, nx):
                        ddx_phi[pp, ii] = lag_ddx_eval(xxb, pp, xxb[ii])
                ddx_phis[nx] = ddx_phi[:,:]
                
            idx: int = 0
            for ii in range(0, nx):
                for jj in range(0, ny):
                    wxi_wyj = wx_wy[ii, jj]
                    for aa in range(0, nth):
                        wth_costh_a: float = wth_costh[aa]
                        for pp in range(0, nx):
                            ddx_phi_pi: float = ddx_phi[pp, ii]
                            
                            alphalist_jqar[idx] = alpha(pp, jj, aa)
                            betalist_jqar[idx]  = beta( ii, jj, aa)
                            
                            vlist_jqar[idx] = dcoeff * wxi_wyj * wth_costh_a \
                                * ddx_phi_pi
                            
                            idx += 1
                            
            delta_jqar: sp._coo.coo_matrix = sp.coo_matrix((vlist_jqar,
                                        (alphalist_jqar, betalist_jqar)),
                                       shape = (cell_ndof, cell_ndof))
            
            cell_mtxs[cell_idx] = delta_ipar + delta_jqar
            
        col_mtx: sp._csr.csr_matrix = sp.block_diag(cell_mtxs, format = "csr")
        
    else:
        col_mtx: sp._csr.csr_matrix = None
        
    return col_mtx
