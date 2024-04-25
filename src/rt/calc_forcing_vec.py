import numpy    as     np
import petsc4py
from   mpi4py   import MPI
from   petsc4py import PETSc
from   time     import perf_counter

import dg.matrix     as mat
import dg.projection as proj
import dg.quadrature as qd
import utils

def calc_forcing_vec(mesh, f, **kwargs):
    return calc_forcing_vec_mpi(mesh, f, **kwargs)

def calc_forcing_vec_seq(mesh, f, **kwargs):
    """
    Create the global vector corresponding to the forcing term f.
    """

    default_kwargs = {'precondition' : False, # Calculate PC matrix
                      'verbose'      : False, # Print info while executing
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
            'Constructing Forcing Vector...\n'
            )
        utils.print_msg(msg)
        
    # Split the problem into parts dependent on size of communicator
    if comm_rank == 0:
        n_global = mesh.get_ndof()
        n_global = MPI_comm.bcast(n_global, root = 0)
        mesh     = MPI_comm.bcast(mesh, root = 0)
    else:
        n_global = None
        n_global = MPI_comm.bcast(n_global, root = 0)
        mesh     = MPI_comm.bcast(mesh, root = 0)
    col_keys = list(sorted(mesh.cols.keys()))
    n_col = len(col_keys)
    p_col_keys = np.array_split(col_keys, comm_size)
    col_keys_local = p_col_keys[comm_rank].astype(np.int32)
    n_local = 0
    for col_key in col_keys_local:
        col = mesh.cols[col_key]
        if col.is_lf:
            [ndof_x, ndof_y] = col.ndofs[:]
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [ndof_th] = cell.ndofs[:]
                    
                    n_local += ndof_x * ndof_y * ndof_th
                    
    n_locals = MPI_comm.allgather(n_local)
    start_idx = int(np.sum(n_locals[:comm_rank]))
    f_vec_local = np.zeros(n_local)
    
    # Get local contribution to forcing vector
    idx = 0
    for col_key in col_keys_local:
        col = mesh.cols[col_key]
        if col.is_lf:
            # Get column information, quadrature weights
            [x0, y0, x1, y1] = col.pos
            [dx, dy]         = [x1 - x0, y1 - y0]
            [ndof_x, ndof_y] = col.ndofs
            
            [xxb, w_x, yyb, w_y, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                      nnodes_y = ndof_y)
            
            xxf = proj.push_forward(x0, x1, xxb)
            yyf = proj.push_forward(y0, y1, yyb)
            
            # Create cell indexing for constructing column forcing vector
            cell_items = sorted(col.cells.items())                
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    # Get cell information, quadrature weights
                    [th0, th1] = cell.pos
                    dth        = th1 - th0
                    [ndof_th]  = cell.ndofs
                    
                    [_, _, _, _, thb, w_th] = qd.quad_xyth(nnodes_th = ndof_th)
                    
                    thf = proj.push_forward(th0, th1, thb)
                    
                    dcoeff = dx * dy * dth / 8.
                    
                    # List of entries, values for constructing the cell mask
                    cell_ndof  = ndof_x * ndof_y * ndof_th
                    f_cell_vec = np.zeros([cell_ndof])
                    for ii in range(0, ndof_x):
                        wx_i = w_x[ii]
                        for jj in range(0, ndof_y):
                            wy_j = w_y[jj]
                            for aa in range(0, ndof_th):
                                wth_a = w_th[aa]
                                f_ija = f(xxf[ii], yyf[jj], thf[aa])
                                
                                f_vec_local[idx] = dcoeff * wx_i * wy_j \
                                    * wth_a * f_ija
                                idx += 1
                                
    f_vec_MPI = PETSc.Vec()
    f_vec_MPI.createMPI(size = n_global, comm = comm)
    for pp in range(0, n_local):
        f_vec_MPI[pp + start_idx] = f_vec_local[pp]
    f_vec_MPI.assemblyBegin()
    f_vec_MPI.assemblyEnd()

    f_vec = MPI_comm.gather(f_vec_MPI.array_r, root = 0)
    if comm_rank == 0:
        f_vec = np.concatenate(f_vec)
    else:
        f_vec = np.zeros([1])
    
    if kwargs['verbose']:
        tf = perf_counter()
        msg = (
            'Constructed Forcing Vector\n' +
            12 * ' '  + 'Time Elapsed: {:8.4f} [s]\n'.format(tf - t0)
        )
        utils.print_msg(msg)
    
    if kwargs['blocking']:        
        MPI_comm.Barrier()
    
    return f_vec

def calc_forcing_vec_mpi(mesh, f, **kwargs):
    """
    Create the global vector corresponding to the forcing term f.
    """

    default_kwargs = {'precondition' : False, # Calculate PC matrix
                      'verbose'      : False, # Print info while executing
                      'blocking'     : True   # Synchronize ranks before exiting
                      } 
    kwargs = {**default_kwargs, **kwargs}
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    PETSc_comm = PETSc.COMM_WORLD
    comm_rank  = PETSc_comm.getRank()
    comm_size  = PETSc_comm.getSize()
    
    if kwargs['verbose']:
        t0 = perf_counter()
        msg = (
            'Constructing Forcing Vector...\n'
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
    v_MPI = PETSc.Vec()
    v_MPI.createMPI(size = n_global, comm = PETSc_comm)
    for col_key in col_keys_local:
        col_vec     = calc_col_vec(mesh, col_key, f, **kwargs)
        col_st_idx  = col_st_idxs[col_key]
        nnz_idxs    = np.where(np.abs(col_vec) > 0)[0].astype(np.int32)
        for idx in nnz_idxs:
            v_MPI[col_st_idx + idx] = col_vec[idx]
    v_MPI.assemblyBegin()
    v_MPI.assemblyEnd()
    
    if kwargs['verbose']:
        tf = perf_counter()
        msg = (
            'Constructed Forcing Vector\n' +
            12 * ' '  + 'Time Elapsed: {:8.4f} [s]\n'.format(tf - t0)
        )
        utils.print_msg(msg)
    
    if kwargs['blocking']:        
        MPI_comm.Barrier()
    
    return v_MPI

def calc_col_vec(mesh, col_key, f, **kwargs):
    
    col = mesh.cols[col_key]
    if col.is_lf:
        # Get column information, quadrature weights
        [x0, y0, x1, y1] = col.pos
        [dx, dy]         = [x1 - x0, y1 - y0]
        [nx, ny] = col.ndofs
        
        [xxb, w_x, yyb, w_y, _, _] = qd.quad_xyth(nnodes_x = nx,
                                                  nnodes_y = ny)
        
        xxf = proj.push_forward(x0, x1, xxb)
        yyf = proj.push_forward(y0, y1, yyb)
        
        # Get size of column vector
        col_ndof = 0
        cell_items = sorted(col.cells.items())                
        for cell_key, cell in cell_items:
            if cell.is_lf:
                [nth]     = cell.ndofs[:]
                col_ndof += nx * ny * nth
                
        col_vec = np.zeros([col_ndof])
        
        idx = 0                
        for cell_key, cell in cell_items:
            if cell.is_lf:
                # Get cell information, quadrature weights
                [th0, th1] = cell.pos
                dth        = th1 - th0
                [nth]      = cell.ndofs[:]
                
                [_, _, _, _, thb, w_th] = qd.quad_xyth(nnodes_th = nth)
                
                thf = proj.push_forward(th0, th1, thb)
                
                dcoeff = dx * dy * dth / 8.
                
                # List of entries, values for constructing the cell mask
                cell_ndof  = nx * ny * nth
                f_cell_vec = np.zeros([cell_ndof])
                for ii in range(0, nx):
                    wx_i = w_x[ii]
                    for jj in range(0, ny):
                        wy_j = w_y[jj]
                        for aa in range(0, nth):
                            wth_a = w_th[aa]
                            f_ija = f(xxf[ii], yyf[jj], thf[aa])
                            
                            col_vec[idx] = dcoeff * wx_i * wy_j * wth_a * f_ija
                            idx += 1
                            
    return col_vec

def calc_forcing_vec_old(mesh, f, **kwargs):
    """
    Create the global vector corresponding to the forcing term f.
    """

    default_kwargs = {'precondition' : False, # Calculate PC matrix
                      'verbose'      : False, # Print info while executing
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
            'Constructing Forcing Vector...\n'
            )
        utils.print_msg(msg)
    
    if comm_rank == 0:
        # Create column indexing for constructing global forcing vector,
        # global solution vector
        col_items         = sorted(mesh.cols.items())
        [ncol, col_idxs]  = mat.get_col_idxs(mesh)
        f_col_vecs        = [None] * ncol # Global vector is a 1-D vector
        
        # Unpack f into a column vectors
        for col_key, col in col_items:
            if col.is_lf:
                # Get column information, quadrature weights
                col_idx          = col_idxs[col_key]
                [x0, y0, x1, y1] = col.pos
                [dx, dy]         = [x1 - x0, y1 - y0]
                [ndof_x, ndof_y] = col.ndofs
                
                [xxb, w_x, yyb, w_y, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                          nnodes_y = ndof_y)
                
                xxf = proj.push_forward(x0, x1, xxb)
                yyf = proj.push_forward(y0, y1, yyb)
                
                # Create cell indexing for constructing column forcing vector
                cell_items         = sorted(col.cells.items())
                [ncell, cell_idxs] = mat.get_cell_idxs(mesh, col_key)
                f_cell_vecs        = [None] * ncell
                # Column forcing vector is a 1-D vector
                
                for cell_key, cell in cell_items:
                    if cell.is_lf:
                        # Get cell information, quadrature weights
                        cell_idx   = cell_idxs[cell_key]
                        [th0, th1] = cell.pos
                        dth        = th1 - th0
                        [ndof_th]  = cell.ndofs
                        
                        beta = mat.get_idx_map(ndof_x, ndof_y, ndof_th)
                        
                        [_, _, _, _, thb, w_th] = qd.quad_xyth(nnodes_th = ndof_th)
                        
                        thf = proj.push_forward(th0, th1, thb)
                        
                        dcoeff = dx * dy * dth / 8.
                        
                        # List of entries, values for constructing the cell mask
                        cell_ndof  = ndof_x * ndof_y * ndof_th
                        f_cell_vec = np.zeros([cell_ndof])
                        for ii in range(0, ndof_x):
                            wx_i = w_x[ii]
                            for jj in range(0, ndof_y):
                                wy_j = w_y[jj]
                                for aa in range(0, ndof_th):
                                    wth_a = w_th[aa]
                                    f_ija = f(xxf[ii], yyf[jj], thf[aa])
                                    
                                    beta_idx = beta(ii, jj, aa)
                                    
                                    f_cell_vec[beta_idx] = dcoeff * wx_i * wy_j \
                                        * wth_a * f_ija
                                    
                        f_cell_vecs[cell_idx] = f_cell_vec
                        
                f_col_vecs[col_idx] = np.concatenate(f_cell_vecs, axis = None)
                
        f_vec = np.concatenate(f_col_vecs, axis = None)
    else:
        f_vec = np.zeros([1])
        
    if kwargs['verbose']:
        tf = perf_counter()
        msg = (
            'Constructed Forcing Vector\n' +
            12 * ' '  + 'Time Elapsed: {:8.4f} [s]\n'.format(tf - t0)
        )
        utils.print_msg(msg)
    
    if kwargs['blocking']:        
        MPI_comm.Barrier()
    
    return f_vec
