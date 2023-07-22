import numpy        as     np
import petsc4py
import scipy.sparse as     sp
from   mpi4py       import MPI
from   petsc4py     import PETSc
from   time         import perf_counter

import utils
from   .get_idxs    import get_idx_map, get_col_idxs, get_cell_idxs

def get_intr_mask(mesh, **kwargs):
    """
    We create the mask in a similar way to creating the matrices -
    build the cell masks to assemble the column masks to assemble the global mask.
    scipy.sparse doesn't work for vectors, so we use a dense representation here.
    """

    default_kwargs = {'verbose'      : False, # Print info while executing
                      'blocking'     : True   # Synchronize ranks before exiting
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
            'Constructing Interior Mask...\n'
            )
        utils.print_msg(msg)

    if comm_rank == 0:
        [ncols, col_idxs] = get_col_idxs(mesh)
        col_masks = [None] * ncols # Global mask is a 1-D vector
        
        col_items = sorted(mesh.cols.items())
        
        for col_key, col in col_items:
            if col.is_lf:
                col_idx = col_idxs[col_key]
                [ndof_x, ndof_y] = col.ndofs
                
                [ncells, cell_idxs] = get_cell_idxs(mesh, col_key)
                cell_masks = [None] * ncells # Column mask is a 1-D vector
                
                cell_items = sorted(col.cells.items())
                
                for cell_key, cell in cell_items:
                    if cell.is_lf:
                        # Get cell information, quadrature weights
                        cell_idx   = cell_idxs[cell_key]
                        [ndof_th]  = cell.ndofs
                        
                        S_quad = cell.quad
                        
                        beta = get_idx_map(ndof_x, ndof_y, ndof_th)
                        
                        # List of entries, values for constructing the cell mask
                        cell_ndof = ndof_x * ndof_y * ndof_th
                        cell_mask = np.ones([cell_ndof], dtype = bool)
                        
                        # Construct the cell mask - the boundary is the inflow
                        # part of the spatial domain boundary
                        if ((col.nhbr_keys[0][0] is None) and # If nhbr, would be [0]
                            (col.nhbr_keys[0][1] is None) and
                            (S_quad == 1 or S_quad == 2)): # 0 => Right
                            for jj in range(0, ndof_y):
                                for aa in range(0, ndof_th):
                                    beta_idx = beta(ndof_x - 1, jj, aa)
                                    cell_mask[beta_idx] = False
                                    
                        if ((col.nhbr_keys[1][0] is None) and
                            (col.nhbr_keys[1][1] is None) and
                            (S_quad == 2 or S_quad == 3)): # 1 => Top
                            for ii in range(0, ndof_x):
                                for aa in range(0, ndof_th):
                                    beta_idx = beta(ii, ndof_y - 1, aa)
                                    cell_mask[beta_idx] = False
                                    
                        if ((col.nhbr_keys[2][0] is None) and
                            (col.nhbr_keys[2][1] is None) and
                            (S_quad == 3 or S_quad == 0)): # 2 => Left
                            for jj in range(0, ndof_y):
                                for aa in range(0, ndof_th):
                                    beta_idx = beta(0, jj, aa)
                                    cell_mask[beta_idx] = False
                                    
                        if ((col.nhbr_keys[3][0] is None) and
                            (col.nhbr_keys[3][1] is None) and
                            (S_quad == 0 or S_quad == 1)): # 3 => Bottom
                            for ii in range(0, ndof_x):
                                for aa in range(0, ndof_th):
                                    beta_idx = beta(ii, 0, aa)
                                    cell_mask[beta_idx] = False
                                    
                        cell_masks[cell_idx] = cell_mask
                        
                col_masks[col_idx] = np.concatenate(cell_masks, axis = None)
                
        global_mask = np.concatenate(col_masks, axis = None)
    else:
        global_mask = 0

    if kwargs['verbose']:
        tf = perf_counter()
        msg = (
            'Constructed Interior Mask\n' +
            12 * ' '  + 'Time Elapsed: {:8.4f} [s]\n'.format(tf - t0)
        )
        utils.print_msg(msg)
    
    if kwargs['blocking']:        
        MPI_comm.Barrier()
        
    return global_mask

def get_bdry_mask(mesh):
    """
    We create the mask in a similar way to creating the matrices -
    build the cell masks to assemble the column masks to assemble the global mask.
    scipy.sparse doesn't work for vectors, so we use a dense representation here.
    """
    
    intr_mask = get_intr_mask(mesh)
    
    return np.invert(intr_mask)
