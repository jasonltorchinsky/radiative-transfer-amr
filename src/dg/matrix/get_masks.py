# Standard Library Imports
from time import perf_counter

# Third-Party Library Imports
import numpy as np
import petsc4py

from mpi4py   import MPI
from petsc4py import PETSc

# Local Library Imports
import consts
import utils

# Relative Imports
from .get_idxs import get_idx_map, get_col_idxs, get_cell_idxs
from ..mesh import Mesh

def get_intr_mask(mesh: Mesh, **kwargs) -> np.ndarray:
    return get_intr_mask_seq(mesh, **kwargs)

def get_intr_mask_seq(mesh: Mesh, **kwargs) -> np.ndarray:
    """
    We create the mask in a similar way to creating the matrices -
    build the cell masks to assemble the column masks to assemble the global mask.
    scipy.sparse doesn"t work for vectors, so we use a dense representation here.
    """

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
        msg: str = ( "Constructing Interior Mask...\n" )
        utils.print_msg(msg)

    if comm_rank == consts.COMM_ROOT:
        [ncols, col_idxs] = get_col_idxs(mesh)
        col_masks: list = [None] * ncols # Global mask is a 1-D vector
        
        col_items = sorted(mesh.cols.items())
        
        for col_key, col in col_items:
            if col.is_lf:
                col_idx: list = col_idxs[col_key]
                [ndof_x, ndof_y] = col.ndofs
                
                [ncells, cell_idxs] = get_cell_idxs(mesh, col_key)
                cell_masks: list = [None] * ncells # Column mask is a 1-D vector
                
                cell_items = sorted(col.cells.items())
                
                for cell_key, cell in cell_items:
                    if cell.is_lf:
                        # Get cell information, quadrature weights
                        cell_idx: int = cell_idxs[cell_key]
                        [ndof_th]     = cell.ndofs
                        
                        S_quadrant: int = cell.quadrant
                        
                        beta = get_idx_map(ndof_x, ndof_y, ndof_th)
                        
                        # List of entries, values for constructing the cell mask
                        cell_ndof: int = ndof_x * ndof_y * ndof_th
                        cell_mask: np.ndarray = np.ones([cell_ndof], dtype = bool)
                        
                        # Construct the cell mask - the boundary is the inflow
                        # part of the spatial domain boundary
                        if ((col.nhbr_keys[0][0] is None) and # If nhbr, would be [0]
                            (col.nhbr_keys[0][1] is None) and
                            (S_quadrant == 1 or S_quadrant == 2)): # 0 => Right
                            for jj in range(0, ndof_y):
                                for aa in range(0, ndof_th):
                                    beta_idx: int = beta(ndof_x - 1, jj, aa)
                                    cell_mask[beta_idx] = False
                                    
                        if ((col.nhbr_keys[1][0] is None) and
                            (col.nhbr_keys[1][1] is None) and
                            (S_quadrant == 2 or S_quadrant == 3)): # 1 => Top
                            for ii in range(0, ndof_x):
                                for aa in range(0, ndof_th):
                                    beta_idx: int = beta(ii, ndof_y - 1, aa)
                                    cell_mask[beta_idx] = False
                                    
                        if ((col.nhbr_keys[2][0] is None) and
                            (col.nhbr_keys[2][1] is None) and
                            (S_quadrant == 3 or S_quadrant == 0)): # 2 => Left
                            for jj in range(0, ndof_y):
                                for aa in range(0, ndof_th):
                                    beta_idx: int = beta(0, jj, aa)
                                    cell_mask[beta_idx] = False
                                    
                        if ((col.nhbr_keys[3][0] is None) and
                            (col.nhbr_keys[3][1] is None) and
                            (S_quadrant == 0 or S_quadrant == 1)): # 3 => Bottom
                            for ii in range(0, ndof_x):
                                for aa in range(0, ndof_th):
                                    beta_idx: int = beta(ii, 0, aa)
                                    cell_mask[beta_idx] = False
                                    
                        cell_masks[cell_idx] = cell_mask
                        
                col_masks[col_idx] = np.concatenate(cell_masks, axis = None)
                
        global_mask: np.ndarray = np.concatenate(col_masks, axis = None)
    else:
        global_mask: int = 0

    if kwargs["verbose"]:
        tf: float = perf_counter()
        msg: str = ( "Constructed Interior Mask\n" +
                     12 * " "  + "Time Elapsed: {:8.4f} [s]\n".format(tf - t0))
        utils.print_msg(msg)
    
    if kwargs["blocking"]:        
        MPI_comm.Barrier()
        
    return global_mask

def get_intr_mask_mpi(mesh, **kwargs):
    """
    We create the mask in a similar way to creating the matrices -
    build the cell masks to assemble the column masks to assemble the global mask.
    scipy.sparse doesn"t work for vectors, so we use a dense representation here.
    """

    default_kwargs = {"verbose"      : False, # Print info while executing
                      "blocking"     : True   # Synchronize ranks before exiting
                      } 
    kwargs = {**default_kwargs, **kwargs}
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    comm      = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()

    if kwargs["verbose"]:
        t0 = perf_counter()
        msg = (
            "Constructing Interior Mask...\n"
            )
        utils.print_msg(msg)

    # Split the problem into parts dependent on size of communicator
    mesh     = MPI_comm.bcast(mesh, root = consts.COMM_ROOT)
    n_global = mesh.get_ndof()
    
    # Split the problem into parts dependent on size of COMM_WORLD.
    col_keys_global = list(sorted(mesh.cols.keys()))
    col_keys_local  = np.array_split(col_keys_global, comm_size)[comm_rank].astype(consts.INT)
    
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
    
    [ncells, cell_idxs] = get_cell_idxs(mesh, col_key)
    cell_masks = [None] * ncells # Column mask is a 1-D vector
    idx = 0
    for col_key in col_keys_local:
        col = mesh.cols[col_key]
        if col.is_lf:
            [ndof_x, ndof_y] = col.ndofs[:]
            
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    # Get cell information, quadrature weights
                    cell_idx   = cell_idxs[cell_key]
                    [ndof_th]  = cell.ndofs
                    
                    S_quadrant = cell.quadrant
                    
                    beta = get_idx_map(ndof_x, ndof_y, ndof_th)
                    
                    # List of entries, values for constructing the cell mask
                    cell_ndof = ndof_x * ndof_y * ndof_th
                    cell_mask = np.ones([cell_ndof], dtype = bool)
                    
                    # Construct the cell mask - the boundary is the inflow
                    # part of the spatial domain boundary
                    if ((col.nhbr_keys[0][0] is None) and # If nhbr, would be [0]
                        (col.nhbr_keys[0][1] is None) and
                        (S_quadrant == 1 or S_quadrant == 2)): # 0 => Right
                        for jj in range(0, ndof_y):
                            for aa in range(0, ndof_th):
                                beta_idx = beta(ndof_x - 1, jj, aa)
                                cell_mask[beta_idx] = False
                                
                    if ((col.nhbr_keys[1][0] is None) and
                        (col.nhbr_keys[1][1] is None) and
                        (S_quadrant == 2 or S_quadrant == 3)): # 1 => Top
                        for ii in range(0, ndof_x):
                            for aa in range(0, ndof_th):
                                beta_idx = beta(ii, ndof_y - 1, aa)
                                cell_mask[beta_idx] = False
                                
                    if ((col.nhbr_keys[2][0] is None) and
                        (col.nhbr_keys[2][1] is None) and
                        (S_quadrant == 3 or S_quadrant == 0)): # 2 => Left
                        for jj in range(0, ndof_y):
                            for aa in range(0, ndof_th):
                                beta_idx = beta(0, jj, aa)
                                cell_mask[beta_idx] = False
                                
                    if ((col.nhbr_keys[3][0] is None) and
                        (col.nhbr_keys[3][1] is None) and
                        (S_quadrant == 0 or S_quadrant == 1)): # 3 => Bottom
                        for ii in range(0, ndof_x):
                            for aa in range(0, ndof_th):
                                beta_idx = beta(ii, 0, aa)
                                cell_mask[beta_idx] = False
                                
                    cell_masks[cell_idx] = cell_mask
                    
                col_masks[col_idx] = np.concatenate(cell_masks, axis = None)
                
        global_mask = np.concatenate(col_masks, axis = None)
    else:
        global_mask = None
    
    global_mask = MPI_comm.bcast(global_mask, root = consts.COMM_ROOT)
    
    if kwargs["verbose"]:
        tf = perf_counter()
        msg = (
            "Constructed Interior Mask\n" +
            12 * " "  + "Time Elapsed: {:8.4f} [s]\n".format(tf - t0)
        )
        utils.print_msg(msg)
    
    if kwargs["blocking"]:        
        MPI_comm.Barrier()
        
    return global_mask

def get_bdry_mask(mesh):
    """
    We create the mask in a similar way to creating the matrices -
    build the cell masks to assemble the column masks to assemble the global mask.
    scipy.sparse doesn"t work for vectors, so we use a dense representation here.
    """
    
    intr_mask = get_intr_mask(mesh)
    
    return np.invert(intr_mask)
