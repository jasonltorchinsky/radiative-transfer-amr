# Standard Library Imports
from time import perf_counter

# Third-Party Library Imports
import numpy as np
import petsc4py
from   mpi4py   import MPI
from   petsc4py import PETSc

# Local Library Imports
import consts
from dg.mesh import Mesh
from dg.mesh.column import Column
from dg.projection import push_forward, pull_back
from dg.quadrature import lag_eval, quad_xyth
import utils

# Relative Imports


def boundary_conditions_vector(self, mesh: Mesh, **kwargs) -> PETSc.Vec:
    """
    Create the global vector corresponding to the boundary conditions.
    
    For now, we are going to do the slow-execution, fast development route of
    having the BCs function handle having the correct values on the boundary.
    """
    
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

    if kwargs["verbose"]:
        t0: float = perf_counter()
        msg: str = ( "Constructing Boundary Conditions Vector...\n" )
        utils.print_msg(msg)
        
    # Share information that is stored on root process
    mesh: Mesh = MPI_comm.bcast(mesh, root = consts.COMM_ROOT)
    n_global: int = mesh.get_ndof()
    
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
        assert( prev_col.is_lf)
        
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
        col_vec: np.ndarray = calc_col_vec(mesh, col_key, self.bcs_dirac)
        col_st_idx: int  = col_st_idxs[col_key]
        nnz_idxs: np.ndarray = np.where(np.abs(col_vec) > 0)[0].astype(consts.INT)
        for idx in nnz_idxs:
            v_MPI[col_st_idx + idx] = col_vec[idx]
    v_MPI.assemblyBegin()
    v_MPI.assemblyEnd()
    
    if kwargs["verbose"]:
        tf: str = perf_counter()
        msg: str = ( "Constructed Boundary Conditions Vector\n" +
                     12 * " "  + "Time Elapsed: {:8.4f} [s]\n".format(tf - t0) )
        utils.print_msg(msg)
    
    if kwargs["blocking"]:        
        MPI_comm.Barrier()
        
    return v_MPI

def calc_col_vec(mesh: Mesh, col_key: int, bcs_dirac: list) -> np.ndarray:
    col: Column = mesh.cols[col_key]
    assert(col.is_lf)

    [bcs, dirac] = bcs_dirac
    
    # Get column information, quadrature weights
    [x0, y0, x1, y1] = col.pos[:]
    [nx, ny] = col.ndofs[:]
    
    [xxb, _, yyb, _, _, _] = quad_xyth(nnodes_x = nx, nnodes_y = ny)
    
    xxf: np.ndarray = push_forward(x0, x1, xxb)
    yyf: np.ndarray = push_forward(y0, y1, yyb)
    
    # Get size of column vector
    col_ndof   = 0
    cell_items = sorted(col.cells.items())                
    for _, cell in cell_items:
        assert(cell.is_lf)
        
        [nth]     = cell.ndofs[:]
        col_ndof += nx * ny * nth
            
    col_vec: np.ndarray = np.zeros([col_ndof])
    
    idx: int = 0
    cell_items: list = sorted(col.cells.items())
    for _, cell in cell_items:
        assert(cell.is_lf)
            
        # Get cell information, quadrature weights
        [th0, th1] = cell.pos[:]
        [nth]      = cell.ndofs[:]
        
        [_, _, _, _, thb, _] = quad_xyth(nnodes_th = nth)
        
        thf: np.ndarray = push_forward(th0, th1, thb)
        
        # If the BCs are a dirac-delta function, we handle
        # them differently
        if any(dirac):
            in_cell: list = [True, True, True]
            if dirac[0] is not None:
                xs_f: float  = dirac[0]
                if (xs_f < x0) or (x1 < xs_f):
                    in_cell[0] = False
                else:
                    xs_b: np.ndarray = pull_back(x0, x1, xs_f)
                    
            if dirac[1] is not None:
                ys_f: float  = dirac[1]
                if (ys_f < y0) or (y1 < ys_f):
                    in_cell[1] = False
                else:
                    ys_b: np.ndarray = pull_back(y0, y1, ys_f)
                    
            if dirac[2] is not None:
                ths_f: float = dirac[2]
                if (ths_f < th0) or (th1 < ths_f):
                    in_cell[2] = False
                else:
                    ths_b: np.ndarray = pull_back(th0, th1, ths_f)
                    
            if any(in_cell):
                for ii in range(0, nx):
                    if ((dirac[0] is not None)
                        and in_cell[0]):
                        phi_i: float = max(0.0, lag_eval(xxb, ii, xs_b))
                        x_i: float   = xs_f
                    else:
                        phi_i: float = 1.
                        x_i: float   = xxf[ii]
                        
                    for jj in range(0, ny):
                        if ((dirac[1] is not None)
                            and in_cell[1]):
                            psi_j: float = max(0.0, lag_eval(yyb, jj, ys_b))
                            y_j: float   = ys_f
                        else:
                            psi_j: float = 1.
                            y_j: float   = yyf[jj]
                            
                        for aa in range(0, nth):
                            if ((dirac[2] is not None)
                                and in_cell[2]):
                                xsi_a: float = max(0.0, lag_eval(thb, aa, ths_b))
                                th_a: float  = ths_f
                            else:
                                xsi_a: float = 1.
                                th_a: float  = thf[aa]
                                
                            bcs_ija: np.ndarray = bcs(x_i, y_j, th_a)
                            
                            col_vec[idx] = bcs_ija * phi_i * psi_j * xsi_a
                            idx += 1
                            
        else:
            for ii in range(0, nx):
                for jj in range(0, ny):
                    for aa in range(0, nth):
                        bcs_ija: np.ndarray = bcs(xxf[ii], yyf[jj], thf[aa])
                        
                        col_vec[idx] = bcs_ija
                        idx += 1
                                
    return col_vec
