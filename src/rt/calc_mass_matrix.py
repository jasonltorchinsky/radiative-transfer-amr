# Standard Library Imports
from time import perf_counter
from typing import Callable

# Third-Party Library Imports
import numpy as np
import petsc4py
import scipy.sparse as sp
from   mpi4py   import MPI
from   petsc4py import PETSc

# Local Library Imports
import consts
from dg.mesh import Mesh
from dg.mesh.column import Column
from dg.projection import push_forward, get_idx_map, get_cell_idxs
from dg.quadrature import lag_ddx_eval, quad_xyth
import utils

# Relative Imports

def calc_mass_matrix(mesh: Mesh, kappa: Callable[[np.ndarray, np.ndarray, np.ndarray], 
                                                 np.ndarray], 
                    **kwargs) -> sp._coo.coo_matrix:
    default_kwargs: dict = {"verbose"      : False, # Print info while executing
                      "blocking"     : True   # Synchronize ranks before exiting
                      }
    kwargs: dict = {**default_kwargs, **kwargs}

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
        
        [nx, ny] = prev_col.ndofs[:]
        
        cell_items: list = sorted(prev_col.cells.items())
        for _, cell in cell_items:
            assert(cell.is_lf)
            
            [ndof_th]  = cell.ndofs[:]
            dof_count += nx * ny * ndof_th
        col_st_idxs[col_key] = dof_count
    col_ndofs[col_keys_global[-1]] = n_global - dof_count
    
    # Create PETSc sparse matrix
    M_MPI: PETSc.Mat = PETSc.Mat()
    M_MPI.createAIJ(size = [n_global, n_global], comm = PETSc_comm)

    # We assemble the column matrices using the block construction of scipy
    for col_key in col_keys_local:
        col_mtx: sp._csr.csr_matrix = calc_col_matrix(mesh, col_key, kappa)
        col_st_idx: list = col_st_idxs[col_key]
        (II, JJ, VV) = sp.find(col_mtx)
        nnz_local: np.ndarray = np.size(II)
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

def calc_col_matrix(mesh: Mesh, col_key: int, kappa: Callable[[np.ndarray, np.ndarray, np.ndarray], 
                                                 np.ndarray]) -> sp._csr.csr_matrix:
    col: Column = mesh.cols[col_key]
    if col.is_lf:
        # Get column information, quadrature weights
        [x0, y0, x1, y1] = col.pos
        [dx, dy] = [x1 - x0, y1 - y0]
        [nx, ny] = col.ndofs
        
        [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = nx, nnodes_y = ny)

        xxf: np.ndarray = push_forward(x0, x1, xxb).reshape([nx, 1])
        yyf: np.ndarray = push_forward(y0, y1, yyb).reshape([1, ny])
        wx: np.ndarray = wx.reshape([nx, 1])
        wy: np.ndarray = wy.reshape([1, ny])
        
        kappa_col: np.ndarray = kappa(xxf, yyf)
        wx_wy_kappa_col: np.ndarray = wx * wy * kappa_col
        
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
            [nth] = cell.ndofs[:]
            [_, _, _, _, _, w_th] = quad_xyth(nnodes_th = nth)
            
            dcoeff: float = dx * dy * dth / 8.
            
            # Indexing from i, j, a to beta
            # Same formula for p, q, r to alpha, but we define alpha
            # anyway for clarity
            #alpha = get_idx_map(nx, ny, nth)
            beta = get_idx_map(nx, ny, nth)
            
            # Set up arrays for delta_ip * delta_ar term
            cell_ndof: int = nx * ny * nth
            betalist: np.ndarray = np.zeros([cell_ndof], dtype = consts.INT) # beta index
            vlist: np.ndarray    = np.zeros([cell_ndof]) # Entry value
            
            # Construct cell matrix
            idx: int = 0
            for ii in range(0, nx):
                for jj in range(0, ny):
                    wx_wy_kappa_ij: float = wx_wy_kappa_col[ii, jj]
                    for aa in range(0, nth):
                        wth_a: float = w_th[aa]
                        
                        # Calculate entry index, value
                        betalist[idx] = beta(ii, jj, aa)
                        
                        vlist[idx] = dcoeff * wth_a * wx_wy_kappa_ij
                        
                        idx += 1
                        
            cell_mtxs[cell_idx] = sp.coo_matrix((vlist, (betalist, betalist)))
                
        col_mtx: sp._csr.csr_matrix = sp.block_diag(cell_mtxs, format = "csr")
        
    else:
        col_mtx: sp._csr.csr_matrix = None
        
    return col_mtx
