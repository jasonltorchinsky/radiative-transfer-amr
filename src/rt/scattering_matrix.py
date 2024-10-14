# Standard Library Imports
from time import perf_counter
from typing import Callable

# Third-Party Library Imports
import numpy as np
import petsc4py
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from   mpi4py   import MPI
from   petsc4py import PETSc

# Local Library Imports
import consts
from dg.matrix import get_idx_map, get_cell_idxs
from dg.mesh import Mesh
from dg.mesh.column import Column
from dg.projection import push_forward
from dg.quadrature import quad_xyth
import utils

def scattering_matrix(self, mesh: Mesh, **kwargs) -> sp._coo.coo_matrix:
    default_kwargs = {"verbose"      : False, # Print info while executing
                      "blocking"     : True   # Synchronize ranks before exiting
                      }
    kwargs = {**default_kwargs, **kwargs}
    
    # Initialize parallel communicators
    mpi_comm: MPI.Intracomm = MPI.COMM_WORLD
    
    if not PETSc.Sys.isInitialized():
        petsc4py.init(comm = mpi_comm)
    petsc_comm: PETSc.Comm = PETSc.COMM_WORLD
    comm_rank: int = petsc_comm.getRank()
    comm_size: int = petsc_comm.getSize()
    
    if kwargs["verbose"]:
        t0: float = perf_counter()
        msg: str = ( "Constructing Scattering Matrix...\n" )
        utils.print_msg(msg)
    
    if comm_rank == consts.COMM_ROOT:
        n_global: int = mesh.get_ndof()
    else:
        n_global: int = None
    n_global: int = mpi_comm.bcast(n_global, root = consts.COMM_ROOT)
    
    # Split the problem into parts dependent on size of COMM_WORLD.
    col_keys_global: list = list(sorted(mesh.cols.keys()))
    col_keys_local: np.ndarray  = np.array_split(col_keys_global, comm_size)[comm_rank].astype(consts.INT)
    
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
    M_MPI.createAIJ(size = [n_global, n_global], comm = petsc_comm)

    # We assemble the column matrices using the block construction of scipy
    for col_key in col_keys_local:
        col_mtx: sp._csr.csr_matrix = calc_col_matrix(mesh, col_key, self.sigma, self.Phi)
        col_st_idx: int = col_st_idxs[col_key]
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
        msg: str = ( "Constructed (Extinction - Scattering) Matrix\n" +
                     12 * " "  + "Time Elapsed: {:8.4f} [s]\n".format(tf - t0) )
        utils.print_msg(msg)
       
    if kwargs["blocking"]:
        mpi_comm.Barrier()
        
    return M_MPI

def calc_col_matrix(mesh: Mesh, col_key: int,
                    sigma: Callable[[np.ndarray, np.ndarray, np.ndarray], 
                                    np.ndarray],
                    Phi: Callable[[np.ndarray, np.ndarray],
                                  np.ndarray]) -> sp._csr.csr_matrix:
    
    col: Column = mesh.cols[col_key]
    if col.is_lf:
        SMALL: float = 1.e1 * consts.EPS

        # Get column information, quadrature weights
        [x0, y0, x1, y1] = col.pos
        [dx, dy] = [x1 - x0, y1 - y0]
        [nx, ny] = col.ndofs
        
        [xxb, wx, yyb, wy, _, _] = quad_xyth(nnodes_x = nx, nnodes_y = ny)
        
        xxf: np.ndarray = push_forward(x0, x1, xxb).reshape([nx, 1])
        wx: np.ndarray  = wx.reshape([nx, 1])
        yyf: np.ndarray = push_forward(y0, y1, yyb).reshape([1, ny])
        wy: np.ndarray  = wy.reshape([1, ny])
        
        # Scattering matrix coefficient
        wx_wy_sigma_h: np.ndarray = wx * wy * sigma(xxf, yyf)
        
        # Create cell indexing for constructing column matrix
        [ncells, cell_idxs] = get_cell_idxs(mesh, col_key)
        cell_mtxs: list = [[None] * ncells for K in range(0, ncells)]
        # Column matrix is dense, and so there are
        # up to ncell**2 non-zero cell scattering matrices
        
        cell_items: list = sorted(col.cells.items())
        
        # _0 refers to cell K in the equations
        # _1 refers to cell K" in the equations
        # Have nested loops because column scattering matrix is dense.
        for cell_key_0, cell_0 in cell_items:
            assert(cell_0.is_lf)
            
            # Get cell information, quadrature weights
            cell_idx_0: int = cell_idxs[cell_key_0]
            [th0_0, th1_0] = cell_0.pos
            dth_0: float = th1_0 - th0_0
            [nth_0] = cell_0.ndofs
            
            [_, _, _, _, thb_0, wth_0] = quad_xyth(nnodes_th = nth_0)
            
            thf_0: np.ndarray = push_forward(th0_0, th1_0, thb_0)
            
            # Indexing from p, q, r to alpha
            alpha = get_idx_map(nx, ny, nth_0)
            
            # Values common to equation for each entry
            dcoeff: float = dx * dy * dth_0 / 8
            
            for cell_key_1, cell_1 in cell_items:
                assert(cell_1.is_lf)
                    
                # Get cell information, quadrature weights
                cell_idx_1: int = cell_idxs[cell_key_1]
                [th0_1, th1_1] = cell_1.pos
                dth_1: float = th1_1 - th0_1
                [nth_1] = cell_1.ndofs
                
                [_, _, _, _, thb_1, wth_1] = quad_xyth(nnodes_th = nth_1)
                thf_1: np.ndarray = push_forward(th0_1, th1_1, thb_1)
                
                Phi_h: np.ndarray = np.zeros([nth_0, nth_1])
                for rr in range(0, nth_0):
                    for aa in range(0, nth_1):
                        Phi_h[rr, aa] = Phi(thf_0[rr], thf_1[aa])
                        
                # List of coordinates, values for
                # constructing cell matrices
                cell_ndof: int = nth_0 * nth_1 * nx * ny
                alphalist: np.ndarray = np.zeros([cell_ndof], dtype = consts.INT)
                betalist: np.ndarray  = np.zeros([cell_ndof], dtype = consts.INT)
                vlist: np.ndarray     = np.zeros([cell_ndof])
                
                # Indexing from i, j, a to beta
                beta = get_idx_map(nx, ny, nth_1)
                
                # Construct cell matrix - 2 cases: K = K", K != K"
                idx: int = 0
                if cell_key_0 != cell_key_1:
                    for ii in range(0, nx):
                        for jj in range(0, ny):
                            wx_wy_sigma_ij: float = wx_wy_sigma_h[ii, jj]
                            if np.abs(wx_wy_sigma_ij) > SMALL:
                                for rr in range(0, nth_0):
                                    wth_0_rr = wth_0[rr]
                                    for aa in range(0, nth_1):
                                        wth_1_aa: float = wth_1[aa]
                                        Phi_ra: float = Phi_h[rr, aa]
                                        
                                        val: float = -dcoeff * (dth_1 / 2.0) \
                                            * wx_wy_sigma_ij * wth_0_rr \
                                            * wth_1_aa * Phi_ra
                                        
                                        if np.abs(val) > SMALL:
                                            # Index of entry
                                            alphalist[idx] = alpha(ii, jj, rr)
                                            betalist[idx]  = beta( ii, jj, aa)
                                            
                                            vlist[idx] = val
                                            idx += 1
                else: # cell_key_0 == cell_key_1
                    for ii in range(0, nx):
                        for jj in range(0, ny):
                            wx_wy_sigma_ij: float = wx_wy_sigma_h[ii, jj]
                            for rr in range(0, nth_0):
                                wth_0_rr: float = wth_0[rr]
                                for aa in range(0, nth_1):
                                    wth_1_aa: float = wth_1[aa]
                                    Phi_ra: float = Phi_h[rr, aa]
                                    
                                    val: float = -dcoeff * (dth_1 / 2.0) \
                                        * wx_wy_sigma_ij * wth_0_rr * wth_1_aa \
                                        * Phi_ra # Scattering term
                                        
                                    if np.abs(val) > SMALL:
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
        col_mtx: sp._csr.csr_matrix = sp.bmat(cell_mtxs, format = "csr")
            
    else:
        col_mtx: sp._csr.csr_matrix = [None, None]
    
    return col_mtx
