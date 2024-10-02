import numpy    as     np
import petsc4py
from   mpi4py   import MPI
from   petsc4py import PETSc
from   time     import perf_counter

import dg.matrix     as mat
import dg.projection as proj
import dg.quadrature as qd
import utils

def calc_bcs_vec(mesh, bcs_dirac, **kwargs):
    return calc_bcs_vec_mpi(mesh, bcs_dirac, **kwargs)

def calc_bcs_vec_seq(mesh, bcs_dirac, **kwargs):
    """
    Create the global vector corresponding to the boundary conditions.
    
    For now, we are going to do the slow-execution, fast development route of
    having the BCs function handle having the correct values on the boundary.
    """
    
    default_kwargs = {"precondition" : False, # Calculate PC matrix
                      "verbose"      : False, # Print info while executing
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
            "Constructing Boundary Conditions Vector...\n"
            )
        utils.print_msg(msg)
        
    if comm_rank == 0:
        [bcs, dirac] = bcs_dirac
        
        intr_mask = mat.get_intr_mask(mesh, blocking = False)
        bdry_mask = np.invert(intr_mask)
        
        # Create column indexing for constructing global bcs vector
        col_items         = sorted(mesh.cols.items())
        [ncol, col_idxs]  = mat.get_col_idxs(mesh)
        bcs_col_vecs      = [None] * ncol # Global vector is a 1-D vector
        
        # Unpack f into a column vectors
        for col_key, col in col_items:
            if col.is_lf:
                # Get column information, quadrature weights
                col_idx          = col_idxs[col_key]
                [x0, y0, x1, y1] = col.pos
                [ndof_x, ndof_y] = col.ndofs
                
                [xxb, _, yyb, _, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                      nnodes_y = ndof_y)
                
                xxf = proj.push_forward(x0, x1, xxb)
                yyf = proj.push_forward(y0, y1, yyb)
                
                # Create cell indexing for constructing column bcs vector
                cell_items         = sorted(col.cells.items())
                [ncell, cell_idxs] = mat.get_cell_idxs(mesh, col_key)
                bcs_cell_vecs      = [None] * ncell # Column forcing vector is a 1-D vector
                
                for cell_key, cell in cell_items:
                    if cell.is_lf:
                        # Get cell information, quadrature weights
                        cell_idx   = cell_idxs[cell_key]
                        [th0, th1] = cell.pos
                        [ndof_th]  = cell.ndofs
                        
                        beta = mat.get_idx_map(ndof_x, ndof_y, ndof_th)
                        
                        [_, _, _, _, thb, _] = qd.quad_xyth(nnodes_th = ndof_th)
                    
                        thf  = proj.push_forward(th0, th1, thb)
                        
                        # List of entries, values for constructing the cell mask
                        cell_ndof  = ndof_x * ndof_y * ndof_th
                        bcs_cell_vec = np.zeros([cell_ndof])
                        
                        # If the BCs are a dirac-delta function, we handle
                        # them differently
                        if any(dirac):
                            in_cell = [True, True, True]
                            if dirac[0] is not None:
                                xs_f  = dirac[0]
                                if (xs_f < x0) or (x1 < xs_f):
                                    in_cell[0] = False
                                else:
                                    xs_b = proj.pull_back(x0, x1, xs_f)
                                    
                            if dirac[1] is not None:
                                ys_f  = dirac[1]
                                if (ys_f < y0) or (y1 < ys_f):
                                    in_cell[1] = False
                                else:
                                    ys_b = proj.pull_back(y0, y1, ys_f)
                                    
                            if dirac[2] is not None:
                                ths_f = dirac[2]
                                if (ths_f < th0) or (th1 < ths_f):
                                    in_cell[2] = False
                                else:
                                    ths_b = proj.pull_back(th0, th1, ths_f)
                                    
                            if any(in_cell):
                                for ii in range(0, ndof_x):
                                    if ((dirac[0] is not None)
                                        and in_cell[0]):
                                        phi_i = max(0.0, qd.lag_eval(xxb, ii, xs_b))
                                        x_i   = xs_f
                                    else:
                                        phi_i = 1.
                                        x_i   = xxf[ii]
                                        
                                    for jj in range(0, ndof_y):
                                        if ((dirac[1] is not None)
                                            and in_cell[1]):
                                            psi_j = max(0.0, qd.lag_eval(yyb, jj, ys_b))
                                            y_j   = ys_f
                                        else:
                                            psi_j = 1.
                                            y_j   = yyf[jj]
                                            
                                        for aa in range(0, ndof_th):
                                            if ((dirac[2] is not None)
                                                and in_cell[2]):
                                                xsi_a = max(0.0, qd.lag_eval(thb, aa, ths_b))
                                                th_a  = ths_f
                                            else:
                                                xsi_a = 1.
                                                th_a  = thf[aa]
                                                
                                            bcs_ija = bcs(x_i, y_j, th_a)
                                            
                                            beta_idx = beta(ii, jj, aa)
                                            
                                            bcs_cell_vec[beta_idx] = bcs_ija \
                                                * phi_i * psi_j * xsi_a
                                            
                        else:
                            for ii in range(0, ndof_x):
                                for jj in range(0, ndof_y):
                                    for aa in range(0, ndof_th):
                                        bcs_ija = bcs(xxf[ii], yyf[jj], thf[aa])
                                        
                                        beta_idx = beta(ii, jj, aa)
                                        
                                        bcs_cell_vec[beta_idx] = bcs_ija
                                        
                        bcs_cell_vecs[cell_idx] = bcs_cell_vec
                        
                bcs_col_vecs[col_idx] = np.concatenate(bcs_cell_vecs, axis = None)
                
        bcs_vec = np.concatenate(bcs_col_vecs, axis = None)
        bcs_vec = bcs_vec[bdry_mask]
    else:
        bcs_vec = np.zeros([1])
        
    if kwargs["verbose"]:
        tf = perf_counter()
        msg = (
            "Constructed Boundary Conditions Vector\n" +
            12 * " "  + "Time Elapsed: {:8.4f} [s]\n".format(tf - t0)
        )
        utils.print_msg(msg)
    
    if kwargs["blocking"]:        
        MPI_comm.Barrier()
    
    return bcs_vec

def calc_bcs_vec_mpi(mesh, bcs_dirac, **kwargs):
    """
    Create the global vector corresponding to the boundary conditions.
    
    For now, we are going to do the slow-execution, fast development route of
    having the BCs function handle having the correct values on the boundary.
    """
    
    default_kwargs = {"precondition" : False, # Calculate PC matrix
                      "verbose"      : False, # Print info while executing
                      "blocking"     : True   # Synchronize ranks before exiting
                      } 
    kwargs = {**default_kwargs, **kwargs}
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    PETSc_comm = PETSc.COMM_WORLD
    comm_rank  = PETSc_comm.getRank()
    comm_size  = PETSc_comm.getSize()

    if kwargs["verbose"]:
        t0 = perf_counter()
        msg = (
            "Constructing Boundary Conditions Vector...\n"
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
        col_vec     = calc_col_vec(mesh, col_key, bcs_dirac, **kwargs)
        col_st_idx  = col_st_idxs[col_key]
        nnz_idxs    = np.where(np.abs(col_vec) > 0)[0].astype(np.int32)
        for idx in nnz_idxs:
            v_MPI[col_st_idx + idx] = col_vec[idx]
    v_MPI.assemblyBegin()
    v_MPI.assemblyEnd()
    
    if kwargs["verbose"]:
        tf = perf_counter()
        msg = (
            "Constructed Boundary Conditions Vector\n" +
            12 * " "  + "Time Elapsed: {:8.4f} [s]\n".format(tf - t0)
        )
        utils.print_msg(msg)
    
    if kwargs["blocking"]:        
        MPI_comm.Barrier()
        
    return v_MPI

def calc_col_vec_new(mesh, col_key, bcs_dirac, **kwargs):
    
    col = mesh.cols[col_key]
    if col.is_lf:
        [bcs, dirac] = bcs_dirac
        
        # Get column information, quadrature weights
        [x0, y0, x1, y1] = col.pos[:]
        [dx, dy]         = [x1 - x0, y1 - y0]
        [nx, ny]         = col.ndofs[:]
        
        [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = nx,
                                              nnodes_y = ny)
        
        xxf = proj.push_forward(x0, x1, xxb)
        yyf = proj.push_forward(y0, y1, yyb)
        wx = wx.reshape([nx, 1, 1])
        wy = wy.reshape([1, ny, 1])
        
        # Construct cell vectors individually
        cell_vecs  = []
        cell_items = sorted(col.cells.items())
        for cell_key, cell in cell_items:
            if cell.is_lf:
                # Get cell information, quadrature weights
                [th0, th1]    = cell.pos[:]
                [dth]         = [th1 - th0]
                [nth]         = cell.ndofs[:]
                cell_bc_vals  = np.zeros([nx, ny, nth])
                
                [_, _, _, _, thb, wth] = qd.quad_xyth(nnodes_th = nth)
                
                thf  = proj.push_forward(th0, th1, thb)
                wth  = wth.reshape([1, 1, nth])
                
                cell_bc_vec = np.zeros([nx * ny * nth])
                
                # If the BCs are a dirac-delta function, we handle
                # them differently
                if any(dirac):
                    in_cell = [True, True, True]
                    if dirac[0] is not None:
                        xs_f  = dirac[0]
                        if (xs_f < x0) or (x1 < xs_f):
                            in_cell[0] = False
                        else:
                            xs_b = proj.pull_back(x0, x1, xs_f)
                            
                    if dirac[1] is not None:
                        ys_f  = dirac[1]
                        if (ys_f < y0) or (y1 < ys_f):
                            in_cell[1] = False
                        else:
                            ys_b = proj.pull_back(y0, y1, ys_f)
                            
                    if dirac[2] is not None:
                        ths_f = dirac[2]
                        if (ths_f < th0) or (th1 < ths_f):
                            in_cell[2] = False
                        else:
                            ths_b = proj.pull_back(th0, th1, ths_f)
                            
                    if any(in_cell):
                        for ii in range(0, nx):
                            if ((dirac[0] is not None)
                                and in_cell[0]):
                                phi_i = max(0.0, qd.lag_eval(xxb, ii, xs_b))
                                x_i   = xs_f
                            else:
                                phi_i = 1.
                                x_i   = xxf[ii]
                                
                            for jj in range(0, ny):
                                if ((dirac[1] is not None)
                                    and in_cell[1]):
                                    psi_j = max(0.0, qd.lag_eval(yyb, jj, ys_b))
                                    y_j   = ys_f
                                else:
                                    psi_j = 1.
                                    y_j   = yyf[jj]
                                    
                                for aa in range(0, nth):
                                    if ((dirac[2] is not None)
                                        and in_cell[2]):
                                        xsi_a = max(0.0, qd.lag_eval(thb, aa, ths_b))
                                        th_a  = ths_f
                                    else:
                                        xsi_a = 1.
                                        th_a  = thf[aa]
                                        
                                    bcs_ija = bcs(x_i, y_j, th_a)
                                    
                                    col_vec[idx] = bcs_ija * phi_i * psi_j * xsi_a
                                    idx         += 1
                                    
                else:
                    for ii in range(0, nx):
                        for jj in range(0, ny):
                            for aa in range(0, nth):
                                bcs_ija = bcs(xxf[ii], yyf[jj], thf[aa])
                                
                                col_vec[idx] = bcs_ija
                                idx         += 1
                                
    return col_vec


def calc_col_vec(mesh, col_key, bcs_dirac, **kwargs):
    
    col = mesh.cols[col_key]
    if col.is_lf:
        [bcs, dirac] = bcs_dirac
        
        # Get column information, quadrature weights
        [x0, y0, x1, y1] = col.pos[:]
        [nx, ny] = col.ndofs[:]
        
        [xxb, _, yyb, _, _, _] = qd.quad_xyth(nnodes_x = nx,
                                              nnodes_y = ny)
        
        xxf = proj.push_forward(x0, x1, xxb)
        yyf = proj.push_forward(y0, y1, yyb)
        
        # Get size of column vector
        col_ndof   = 0
        cell_items = sorted(col.cells.items())                
        for cell_key, cell in cell_items:
            if cell.is_lf:
                [nth]     = cell.ndofs[:]
                col_ndof += nx * ny * nth
                
        col_vec = np.zeros([col_ndof])
        
        idx = 0
        [ncell, cell_idxs] = mat.get_cell_idxs(mesh, col_key)
        cell_items         = sorted(col.cells.items())
        for cell_key, cell in cell_items:
            if cell.is_lf:
                # Get cell information, quadrature weights
                cell_idx   = cell_idxs[cell_key]
                [th0, th1] = cell.pos[:]
                [nth]      = cell.ndofs[:]
                
                beta = mat.get_idx_map(nx, ny, nth)
                
                [_, _, _, _, thb, _] = qd.quad_xyth(nnodes_th = nth)
                
                thf  = proj.push_forward(th0, th1, thb)
                
                # List of entries, values for constructing the cell mask
                cell_ndof  = nx * ny * nth
                bcs_cell_vec = np.zeros([cell_ndof])
                
                # If the BCs are a dirac-delta function, we handle
                # them differently
                if any(dirac):
                    in_cell = [True, True, True]
                    if dirac[0] is not None:
                        xs_f  = dirac[0]
                        if (xs_f < x0) or (x1 < xs_f):
                            in_cell[0] = False
                        else:
                            xs_b = proj.pull_back(x0, x1, xs_f)
                            
                    if dirac[1] is not None:
                        ys_f  = dirac[1]
                        if (ys_f < y0) or (y1 < ys_f):
                            in_cell[1] = False
                        else:
                            ys_b = proj.pull_back(y0, y1, ys_f)
                            
                    if dirac[2] is not None:
                        ths_f = dirac[2]
                        if (ths_f < th0) or (th1 < ths_f):
                            in_cell[2] = False
                        else:
                            ths_b = proj.pull_back(th0, th1, ths_f)
                            
                    if any(in_cell):
                        for ii in range(0, nx):
                            if ((dirac[0] is not None)
                                and in_cell[0]):
                                phi_i = max(0.0, qd.lag_eval(xxb, ii, xs_b))
                                x_i   = xs_f
                            else:
                                phi_i = 1.
                                x_i   = xxf[ii]
                                
                            for jj in range(0, ny):
                                if ((dirac[1] is not None)
                                    and in_cell[1]):
                                    psi_j = max(0.0, qd.lag_eval(yyb, jj, ys_b))
                                    y_j   = ys_f
                                else:
                                    psi_j = 1.
                                    y_j   = yyf[jj]
                                    
                                for aa in range(0, nth):
                                    if ((dirac[2] is not None)
                                        and in_cell[2]):
                                        xsi_a = max(0.0, qd.lag_eval(thb, aa, ths_b))
                                        th_a  = ths_f
                                    else:
                                        xsi_a = 1.
                                        th_a  = thf[aa]
                                        
                                    bcs_ija = bcs(x_i, y_j, th_a)
                                    
                                    col_vec[idx] = bcs_ija * phi_i * psi_j * xsi_a
                                    idx         += 1
                                    
                else:
                    for ii in range(0, nx):
                        for jj in range(0, ny):
                            for aa in range(0, nth):
                                bcs_ija = bcs(xxf[ii], yyf[jj], thf[aa])
                                
                                col_vec[idx] = bcs_ija
                                idx         += 1
                                
    return col_vec
