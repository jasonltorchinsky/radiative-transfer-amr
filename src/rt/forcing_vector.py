# Standard Library Imports
from time import perf_counter
from typing import Callable

# Third-Party Library Imports
import numpy        as np
import petsc4py
from   mpi4py       import MPI
from   petsc4py     import PETSc

# Local Library Imports
import consts
from dg.mesh import Mesh
from dg.mesh.column import Column
from dg.matrix import get_col_idxs, get_cell_idxs
from dg.projection import push_forward
from dg.quadrature import quad_xyth
import utils

# Relative Imports

def forcing_vector(self, mesh: Mesh, **kwargs) -> np.ndarray:
    """
    Create the global vector corresponding to the forcing term f.
    """

    default_kwargs: dict = {"precondition" : False, # Calculate PC matrix
                      "verbose"      : False, # Print info while executing
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
        msg: str = ( "Constructing Forcing Vector...\n" )
        utils.print_msg(msg)
        
    # Share information that is stored on root process
    if comm_rank == 0:
        n_global: int = mesh.get_ndof()
    else:
        n_global: int = None
    n_global: int = MPI_comm.bcast(n_global, root = 0)
    
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
        
        [nx, ny] = prev_col.ndofs[:]
        
        cell_items: list = sorted(prev_col.cells.items())
        for _, cell in cell_items:
            assert(cell.is_lf)
                
            [ndof_th]  = cell.ndofs[:]
            dof_count += nx * ny * ndof_th
        col_st_idxs[col_key] = dof_count
    col_ndofs[col_keys_global[-1]] = n_global - dof_count
    
    # Create PETSc sparse matrix
    v_MPI: PETSc.Vec = PETSc.Vec()
    v_MPI.createMPI(size = n_global, comm = PETSc_comm)
    for col_key in col_keys_local:
        col_vec: np.ndarray = calc_col_vec(mesh, col_key, self.f)
        col_st_idx: int = col_st_idxs[col_key]
        nnz_idxs: np.ndarray = np.where(np.abs(col_vec) > 0)[0].astype(consts.INT)
        for idx in nnz_idxs:
            v_MPI[col_st_idx + idx] = col_vec[idx]
    v_MPI.assemblyBegin()
    v_MPI.assemblyEnd()
    
    if kwargs["verbose"]:
        tf: float = perf_counter()
        msg: str = ( "Constructed Forcing Vector\n" +
                     12 * " "  + "Time Elapsed: {:8.4f} [s]\n".format(tf - t0) )
        utils.print_msg(msg)
    
    if kwargs["blocking"]:        
        MPI_comm.Barrier()
    
    return v_MPI

def calc_col_vec(mesh: Mesh, col_key: int, f: Callable[[np.ndarray, np.ndarray, np.ndarray],
                                                       np.ndarray]) -> np.ndarray:
    col: Column = mesh.cols[col_key]
    assert(col.is_lf)
    # Get column information, quadrature weights
    [x0, y0, x1, y1] = col.pos
    [dx, dy] = [x1 - x0, y1 - y0]
    [nx, ny] = col.ndofs
    
    [xxb, w_x, yyb, w_y, _, _] = quad_xyth(nnodes_x = nx, nnodes_y = ny)
    
    xxf: np.ndarray = push_forward(x0, x1, xxb)
    yyf: np.ndarray = push_forward(y0, y1, yyb)
    
    # Get size of column vector
    col_ndof: int = 0
    cell_items: list = sorted(col.cells.items())                
    for _, cell in cell_items:
        assert(cell.is_lf)
            
        [nth]     = cell.ndofs[:]
        col_ndof += nx * ny * nth
            
    col_vec: np.ndarray = np.zeros([col_ndof])
    
    idx: int = 0                
    for _, cell in cell_items:
        assert(cell.is_lf)
        
        # Get cell information, quadrature weights
        [th0, th1] = cell.pos
        dth: float = th1 - th0
        [nth] = cell.ndofs[:]
        
        [_, _, _, _, thb, w_th] = quad_xyth(nnodes_th = nth)
        
        thf: np.ndarray = push_forward(th0, th1, thb)
        
        dcoeff: float = dx * dy * dth / 8.
        
        # List of entries, values for constructing the cell mask
        cell_ndof: int  = nx * ny * nth
        for ii in range(0, nx):
            wx_i: float = w_x[ii]
            for jj in range(0, ny):
                wy_j: float = w_y[jj]
                for aa in range(0, nth):
                    wth_a: float = w_th[aa]
                    f_ija: float = f(xxf[ii], yyf[jj], thf[aa])
                    
                    col_vec[idx] = dcoeff * wx_i * wy_j * wth_a * f_ija
                    idx += 1
                            
    return col_vec