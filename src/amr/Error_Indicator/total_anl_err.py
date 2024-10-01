import numpy           as np
import os
import petsc4py
import scipy.integrate as integrate
from   mpi4py          import MPI
from   petsc4py        import PETSc

import dg.mesh       as ji_mesh
import dg.projection as proj
import dg.quadrature as qd

intg_u2  = {}
phi_mtxs = {}
psi_mtxs = {}
xsi_mtxs = {}

mesh_hr     = None
u_hr        = None
intg_u_hr_2 = None

def total_anl_err(mesh, num_sol, anl_sol, **kwargs):
    #return total_anl_err_2(mesh, num_sol, anl_sol, **kwargs)
    if kwargs['ang_res_offset'] == 0 and kwargs['spt_res_offset'] == 0:
        return total_anl_err_lr(mesh, num_sol, anl_sol, **kwargs)
    else:
        return total_anl_err_hr(mesh, num_sol, anl_sol, **kwargs)

def total_anl_err_2(mesh, num_sol, anl_sol, **kwargs):
    """
    Construct a single high-resolution solution and save it to file for later
    reuse. Compare all numerical solutions against it.
    """
    default_kwargs = { 'ndof_x_hr'  : 6, # x-DoFs for hi-res solution
                       'ndof_y_hr'  : 6, # y-DOFs for hi-res solution
                       'ndof_th_hr' : 6, # th-DoFs for hi-res solution
                       'nref_spt'   : 3, # Number of spatial h-refs
                       'nref_ang'   : 2, # Number of angular h-refs
                       'verbose'    : False, # Print info while executing
                       'blocking'   : True,  # Synchronize ranks before exiting
                       'file_path'  : '',    # Path for saving files
                       'u_hr_file_name' : 'u_hr.npy' # File name for hr-soln
                      }
    kwargs = {**default_kwargs, **kwargs}
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    comm      = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()

    # Get the high-resolution mesh
    global mesh_hr
    if mesh_hr is None:
        if comm_rank == 0:
            [nx_hr, ny_hr, nth_hr] = [kwargs['ndof_x_hr'], kwargs['ndof_y_hr'],
                                      kwargs['ndof_th_hr']]
            mesh_hr = ji_mesh.Mesh(Ls    = mesh.Ls[:],
                                   pbcs  = mesh.pbcs[:],
                                   ndofs = [nx_hr, ny_hr, nth_hr],
                                   has_th = mesh.has_th
                                   )
            for _ in range(0, kwargs['nref_ang']):
                mesh_hr.ref_mesh(kind = 'ang', form = 'h')
            for _ in range(0, kwargs['nref_spt']):
                mesh_hr.ref_mesh(kind = 'spt', form = 'h')
            # Plot the mesh
            file_name = 'mesh_hr.png'
            file_path = os.path.join(kwargs['file_path'], file_name)
            ji_mesh.utils.gen_mesh_plot(mesh, trial, file_path, blocking = False)
            
            file_name = 'mesh_hr_p.png'
            file_path = os.path.join(kwargs['file_path'], file_name)
            ji_mesh.utils.gen_mesh_plot_p(mesh, trial, file_path, blocking = False)
            
        else:
            mesh_hr = None
        mesh_hr = MPI_comm.bcast(mesh_hr, root = 0)
    mesh    = MPI_comm.bcast(mesh, root = 0)
    uh_proj = num_sol
    uh_proj = MPI_comm.bcast(uh_proj, root = 0)
    
    # Get the high-resolution solution
    global u_hr
    if u_hr is None:
        u_hr_file_path = os.path.join(kwargs['file_path'],
                                       kwargs['u_hr_file_name'])
        if not os.path.isfile(u_hr_file_path): # If not saved to file yet
            # Get the solution and save it to file
            u_hr = proj.Projection(mesh_hr, anl_sol)
            if comm_rank == 0:
                u_hr_vec = u_hr.to_vector()
                np.save(u_hr_file_path, u_hr_vec)
            MPI_comm.barrier()
            # Share the high-resolution solution across processes
            u_hr = MPI_comm.bcast(u_hr, root = 0)
            
            # If it's not saved to file, assume we need to plot it
            if comm_rank == 0:
                if kwargs['verbose']:
                    msg = (
                        'Plotting high-resolution solution...\n'
                    )
                    utils.print_msg(msg, blocking = False)
                    t0 = time.perf_counter()
                    
                # Plot high-resolution solution for posterity
                file_name = 'u_hr_th.png'
                file_path = os.path.join(kwargs['file_path'], file_name)
                proj.utils.plot_th(mesh_hr, u_hr, file_name = file_path)
                
                file_name = 'u_hr_xy.png'
                file_path = os.path.join(kwargs['file_path'], file_name)
                proj.utils.plot_xy(mesh_hr, u_hr, file_name = file_path)
                
                file_name = 'u_hr_xth.png'
                file_path = os.path.join(kwargs['file_path'], file_name)
                proj.utils.plot_xth(mesh_hr, u_hr, file_name = file_path)
                
                file_name = 'u_hr_yth.png'
                file_path = os.path.join(kwargs['file_path'], file_name)
                proj.utils.plot_yth(mesh_hr, u_hr, file_name = file_path)
                
                file_name = 'u_hr_xyth.png'
                file_path = os.path.join(kwargs['file_path'], file_name)
                proj.utils.plot_xyth(mesh_hr, u_hr, file_name = file_path)
                
                if kwargs['verbose']:
                    tf = time.perf_counter()
                    dt = tf - t0
                    msg = (
                        'High-resolution solution plotted!\n' +
                        12 * ' ' + 'Time Elapsed: {:8.4f} [s]\n'.format(dt)
                    )
                    utils.print_msg(msg, blocking = False)
        else: # If saved to file, read it from there
            u_hr_vec = np.load(u_hr_file_path)
            u_hr     = proj.to_projection(mesh_hr, u_hr_vec)
            
    # Integrate the square of the high-resolution solution for relative error
    global intg_u_hr_2
    if intg_u_hr_2 is None:
        if kwargs['verbose']:
            msg = (
                'Integrating high-resolution solution...\n'
            )
            utils.print_msg(msg, blocking = False)
            t0 = time.perf_counter()
            
        # Split the problem into parts dependent on size of COMM_WORLD.
        col_keys_hr_global = list(sorted(mesh_hr.cols.keys()))
        col_keys_hr_local  = np.array_split(col_keys_hr_global, comm_size)[comm_rank].astype(np.int32)
        
        local_intg_u_hr_2 = 0.
        # Integrate high-resolution solution
        for col_key_hr in col_keys_hr_local:
            col_hr = mesh_hr.cols[col_key_hr]
            if col_hr.is_lf:
                [nx_hr, ny_hr]   = col_hr.ndofs[:]
                [x0, y0, x1, y1] = col_hr.pos[:]
                [dx, dy]         = [x1 - x0, y1 - y0]
                [_, wx_hr, _, wy_hr, _, _] = qd.quad_xyth(nnodes_x = nx_hr,
                                                          nnodes_y = ny_hr)
                wx_hr = wx_hr.reshape([nx_hr, 1, 1])
                wy_hr = wy_hr.reshape([1, ny_hr, 1])
                
                dcoeff = (dx * dy) / 4.
                
                cell_items_hr = sorted(col_hr.cells.items())
                for cell_key_hr, cell_hr in cell_items_hr:
                    if cell_hr.is_lf:
                        [nth_hr]   = cell_hr.ndofs[:]
                        [th0, th1] = cell_hr.pos[:]
                        dth        = th1 - th0
                        
                        [_, _, _, _, _, wth_hr] = qd.quad_xyth(nnodes_th = nth_hr)
                        
                        wth_hr = wth_hr.reshape([1, 1, nth_hr])
                        
                        u_hr_cell = u_hr.cols[col_key_hr].cells[cell_key_hr].vals[:, :, :]
                        
                        local_intg_u_hr_2 += dcoeff * (dth / 2.) * np.sum(wx_hr * wy_hr * wth_hr * (u_hr_cell)**2)
        intg_u_hr_2 = MPI_comm.allreduce(local_intg_u_hr_2)
        if kwargs['verbose']:
            tf = time.perf_counter()
            dt = tf - t0
            msg = (
                'High-resolution solution integrated!\n' +
                12 * ' ' + 'Time Elapsed: {:8.4f} [s]\n'.format(dt)
            )
            utils.print_msg(msg, blocking = False)

    # Calculate the error
    # Embed the low-res soln in the hi-res mesh and get the L2-error
    if kwargs['verbose']:
        msg = (
            'Calculating high-resolution error...\n'
        )
        utils.print_msg(msg, blocking = False)
        t0 = time.perf_counter()
        
    # Split the problem into parts dependent on size of COMM_WORLD.
    col_keys_global = list(sorted(mesh.cols.keys()))
    col_keys_local  = np.array_split(col_keys_global, comm_size)[comm_rank].astype(np.int32)

    col_keys_hr_global = list(sorted(mesh_hr.cols.keys()))
    local_err = 0.
    for col_key in col_keys_local:
        col = mesh.cols[col_key]
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy]         = [x1 - x0, y1 - y0]
            [nx, ny]         = col.ndofs[:]
            
            # Find the column of the hi-res mesh that this column is in
            [x_mid, y_mid] = [(x0 + x1) / 2., (y0 + y1) / 2.]
            for col_key_hr in col_keys_hr_global:
                col_hr = mesh_hr.cols[col_key_hr]
                if col_hr.is_lf:
                    [x0_hr, y0_hr, x1_hr, y1_hr] = col_hr.pos[:]
                    if ((x0_hr <= x_mid) and (x_mid <= x1_hr)
                        and (y0_hr <= y_mid) and (y_mid <= y1_hr)):
                        break
            [nx_hr, ny_hr] = col_hr.ndofs[:]
            
            # Get the necessary spatial embedding matrices
            phi_mtx_lr = np.zeros([nx, nx_hr])
            phi_mtx_hr = np.zeros([nx_hr, nx_hr])
            [xxb,    _,     _, _, _, _] = qd.quad_xyth(nnodes_x = nx)
            [xxb_hr, wx_hr, _, _, _, _] = qd.quad_xyth(nnodes_x = nx_hr)
            wx_hr = wx_hr.reshape([nx_hr, 1, 1])
            
            xxf_hr_lr = proj.push_forward(x0, x1, xxb_hr)
            xxb_hr_lr = proj.pull_back(x0_hr, x1_hr, xxf_hr_lr)
            for ii in range(0, nx):
                for pp_p in range(0, nx_hr):
                    phi_mtx_lr[ii, pp_p] = qd.lag_eval(xxb, ii, xxb_hr[pp_p])
            for pp in range(0, nx_hr):
                for pp_p in range(0, nx_hr):
                    phi_mtx_hr[pp, pp_p] = qd.lag_eval(xxb_hr, pp, xxb_hr_lr[pp_p])
                    
            psi_mtx_lr = np.zeros([ny, ny_hr])
            psi_mtx_hr = np.zeros([ny_hr, ny_hr])
            [_, _, yyb,    _,     _, _] = qd.quad_xyth(nnodes_y = ny)
            [_, _, yyb_hr, wy_hr, _, _] = qd.quad_xyth(nnodes_y = ny_hr)
            wy_hr = wy_hr.reshape([1, ny_hr, 1])
            
            yyf_hr_lr = proj.push_forward(y0, y1, yyb_hr)
            yyb_hr_lr = proj.pull_back(y0_hr, y1_hr, yyf_hr_lr)
            for jj in range(0, ny):
                for qq_p in range(0, ny_hr):
                    psi_mtx_lr[jj, qq_p] = qd.lag_eval(yyb, jj, yyb_hr[qq_p])
            for qq in range(0, ny_hr):
                for qq_p in range(0, ny_hr):
                    psi_mtx_hr[qq, qq_p] = qd.lag_eval(yyb_hr, qq, yyb_hr_lr[qq_p])
                    
            cell_items = sorted(list(col.cells.items()))
            cell_hr_items = sorted(list(col_hr.cells.items()))
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, th1] = cell.pos[:]
                    [dth]      = [th1 - th0]
                    [nth]      = cell.ndofs[:]
                    
                    # Find the cell of the hi-res mesh that this cell is in
                    th_mid = (th0 + th1) / 2.
                    for cell_key_hr, cell_hr in cell_hr_items:
                        if cell_hr.is_lf:
                            [th0_hr, th1_hr] = cell_hr.pos[:]
                            if ((th0_hr <= th_mid) and (th_mid <= th1_hr)):
                                break
                    [nth_hr] = cell_hr.ndofs[:]
                    # Get the necessary angular embedding matrices
                    xsi_mtx_lr = np.zeros([nth, nth_hr])
                    xsi_mtx_hr = np.zeros([nth_hr, nth_hr])
                    [_, _, _, _, thb,    _]      = qd.quad_xyth(nnodes_th = nth)
                    [_, _, _, _, thb_hr, wth_hr] = qd.quad_xyth(nnodes_th = nth_hr)
                    wth_hr = wth_hr.reshape([1, 1, nth_hr])
                    
                    thf_hr_lr = proj.push_forward(th0, th1, thb_hr)
                    thb_hr_lr = proj.pull_back(th0_hr, th1_hr, thf_hr_lr)
                    for aa in range(0, nth):
                        for rr_p in range(0, nth_hr):
                            xsi_mtx_lr[aa, rr_p] = qd.lag_eval(thb, aa, thb_hr[rr_p])
                    for rr in range(0, nth_hr):
                        for rr_p in range(0, nth_hr):
                            xsi_mtx_hr[rr, rr_p] = qd.lag_eval(thb_hr, rr, thb_hr_lr[rr_p])
                    # Get the hi-, lo-res solutions on the hi-res mesh of the correct interval size
                    u_hr_cell = u_hr.cols[col_key_hr].cells[cell_key_hr].vals[:,:,:]
                    u_hr_cell_proj = np.zeros([nx_hr, ny_hr, nth_hr])
                    for pp_p in range(0, nx_hr):
                        for qq_p in range(0, ny_hr):
                            for rr_p in range(0, nth_hr):
                                for pp in range(0, nx_hr):
                                    phi_pp_p = phi_mtx_hr[pp, pp_p]
                                    for qq in range(0, ny_hr):
                                        psi_qq_p = psi_mtx_hr[qq, qq_p]
                                        for rr in range(0, nth_hr):
                                            xsi_rr_p = xsi_mtx_hr[rr, rr_p]
                                            
                                            u_hr_cell_proj[pp_p, qq_p, rr_p] += \
                                                u_hr_cell[pp, qq, rr] \
                                                * phi_pp_p * psi_qq_p * xsi_rr_p
                                
                    uh_lr_cell = uh_proj.cols[col_key].cells[cell_key].vals[:,:,:]
                    uh_lr_cell_proj = np.zeros([nx_hr, ny_hr, nth_hr])
                    for pp_p in range(0, nx_hr):
                        for qq_p in range(0, ny_hr):
                            for rr_p in range(0, nth_hr):
                                for ii in range(0, nx):
                                    phi_ip_p = phi_mtx_lr[ii, pp_p]
                                    for jj in range(0, ny):
                                        psi_jq_p = psi_mtx_lr[jj, qq_p]
                                        for aa in range(0, nth):
                                            xsi_ar_p = xsi_mtx_lr[aa, rr_p]
                                            
                                            uh_lr_cell_proj[pp_p, qq_p, rr_p] += \
                                                uh_lr_cell[ii, jj, aa] \
                                                * phi_ip_p * psi_jq_p * xsi_ar_p
                                            
                    local_err += (dx * dy * dth / 8.) * np.sum(wx_hr * wy_hr * wth_hr * (u_hr_cell_proj - uh_lr_cell_proj)**2)
    err = MPI_comm.allreduce(local_err)
    err = np.sqrt(err / intg_u_hr_2)
    if kwargs['verbose']:
        tf = time.perf_counter()
        dt = tf - t0
        msg = (
            'High-resolution error obtained!\n' +
            12 * ' ' + 'Time Elapsed: {:8.4f} [s]\n'.format(dt)
        )
        utils.print_msg(msg, blocking = False)
        
    if kwargs['blocking']:
        MPI_comm.Barrier()
        
    return err


def total_anl_err_lr(mesh, num_sol, anl_sol, **kwargs):
    """
    Calculate the L2-error by cell (and column), weighted to be the relative error.
    """
    
    default_kwargs = {'ang_res_offset' : 0,   # Factor to add to angular DoFs
                      'spt_res_offset' : 0,   # Factor to add to spatial DoFs
                      'key'        : ' ', # Key to hold intg_u2
                      'blocking'   : True # Synchronize ranks before exiting
                      }
    kwargs = {**default_kwargs, **kwargs}
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    PETSc_comm = PETSc.COMM_WORLD
    comm_rank  = PETSc_comm.getRank()
    comm_size  = PETSc_comm.getSize()
    
    # Share information that is stored on root process
    mesh     = MPI_comm.bcast(mesh, root = 0)
    num_sol  = MPI_comm.bcast(num_sol, root = 0)
    n_global = mesh.get_ndof()
    
    # Split the problem into parts dependent on size of COMM_WORLD.
    col_keys_global = list(sorted(mesh.cols.keys()))
    col_keys_local  = np.array_split(col_keys_global, comm_size)[comm_rank].astype(np.int32)
        
    # Integrate the square of the analytic solution here
    global intg_u2
    if kwargs['key'] not in intg_u2.keys():
        [Lx, Ly] = mesh.Ls[:]
        [intg_u2[kwargs['key']], _] = integrate.nquad(lambda x, y, th: (anl_sol(x, y, th))**2,
                                                      [[0, Lx], [0, Ly], [0, 2. * np.pi]])
        
    local_err = 0.
    # Calculate the errors
    for col_key in col_keys_local:
        col = mesh.cols[col_key]
        if col.is_lf:
            # Column information for quadrature
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy]         = [x1 - x0, y1 - y0]
            [ndof_x, ndof_y] = col.ndofs[:]
            
            [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                    nnodes_y = ndof_y)
            
            xxf = proj.push_forward(x0, x1, xxb).reshape([ndof_x, 1, 1])
            yyf = proj.push_forward(y0, y1, yyb).reshape([1, ndof_y, 1])
            
            wx = wx.reshape([ndof_x, 1, 1])
            wy = wy.reshape([1, ndof_y, 1])
            
            # Loop through cells to calculate error
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    # Cell information for quadrature
                    [th0, th1] = cell.pos[:]
                    dth = th1 - th0
                    [ndof_th]  = cell.ndofs[:]
                    
                    [_, _, _, _, thb, wth] = qd.quad_xyth(nnodes_th = ndof_th)
                    
                    thf = proj.push_forward(th0, th1, thb).reshape([1, 1, ndof_th])
                    wth = wth.reshape([1, 1, ndof_th])
                    
                    # Calculate error
                    u_cell  = anl_sol(xxf, yyf, thf)
                    uh_cell = num_sol.cols[col_key].cells[cell_key].vals[:, :, :]
                    
                    local_err += (dx * dy * dth / 8.) * np.sum(wx * wy * wth * (u_cell - uh_cell)**2)
                    
    err = MPI_comm.allreduce(local_err)
    err = np.sqrt(err/intg_u2[kwargs['key']])
    
    if kwargs['blocking']:
        MPI_comm.Barrier()

    return err

def total_anl_err_hr(mesh, num_sol, anl_sol, **kwargs):
    """
    Calculate the L2-error by cell (and column), weighted to be the relative error.
    """
    
    default_kwargs = {'ref_kind'   : 'all',
                      'ang_res_offset' : 1,   # Factor to add to angular DoFs
                      'spt_res_offset' : 1,   # Factor to add to spatial DoFs
                      'key'        : ' ', # Key to hold intg_u2
                      'blocking'   : True # Synchronize ranks before exiting
                      }
    kwargs = {**default_kwargs, **kwargs}
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    PETSc_comm = PETSc.COMM_WORLD
    comm_rank  = PETSc_comm.getRank()
    comm_size  = PETSc_comm.getSize()
    
    # Share information that is stored on root process
    mesh     = MPI_comm.bcast(mesh, root = 0)
    num_sol  = MPI_comm.bcast(num_sol, root = 0)
    n_global = mesh.get_ndof()
    
    # Split the problem into parts dependent on size of COMM_WORLD.
    col_keys_global = list(sorted(mesh.cols.keys()))
    col_keys_local  = np.array_split(col_keys_global, comm_size)[comm_rank].astype(np.int32)
    
    ang_res_offset = kwargs['ang_res_offset']
    spt_res_offset = kwargs['spt_res_offset']
    
    # Integrate the square of the analytic solution here
    global intg_u2
    if kwargs['key'] not in intg_u2.keys():
        [Lx, Ly] = mesh.Ls[:]
        [intg_u2[kwargs['key']], _] = integrate.nquad(lambda x, y, th: (anl_sol(x, y, th))**2,
                                                      [[0, Lx], [0, Ly], [0, 2. * np.pi]])
        
    local_err = 0.
    # Calculate the errors
    for col_key in col_keys_local:
        col = mesh.cols[col_key]
        if col.is_lf:
            # Column information for quadrature
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy]         = [x1 - x0, y1 - y0]
            [nx, ny]         = col.ndofs[:]
            
            [xxb, _, yyb, _, _, _] = qd.quad_xyth(nnodes_x = nx,
                                                  nnodes_y = ny)
            
            if kwargs['ref_kind'] in ('spt', 'all'):
                [nx_hr, ny_hr] = [int(spt_res_offset + nx), int(spt_res_offset * ny)]
            else:
                [nx_hr, ny_hr] = [nx, ny]
            
            [xxb_hr, wx_hr, yyb_hr, wy_hr, _, _] = qd.quad_xyth(nnodes_x = nx_hr,
                                                                nnodes_y = ny_hr)
            
            xxf_hr = proj.push_forward(x0, x1, xxb_hr).reshape([nx_hr, 1, 1])
            yyf_hr = proj.push_forward(y0, y1, yyb_hr).reshape([1, ny_hr, 1])
            
            wx_hr = wx_hr.reshape([nx_hr, 1, 1])
            wy_hr = wy_hr.reshape([1, ny_hr, 1])
            
            if (nx, nx_hr) in phi_mtxs.keys():
                phi_mtx = phi_mtxs[(nx, nx_hr)]
            else:
                phi_mtx = np.zeros([nx, nx_hr])
                for ii in range(0, nx):
                    for pp in range(0, nx_hr):
                        phi_mtx[ii, pp] = qd.lag_eval(xxb, ii, xxb_hr[pp])
                phi_mtxs[(nx, nx_hr)] = phi_mtx
                
            if (ny, ny_hr) in psi_mtxs.keys():
                psi_mtx = psi_mtxs[(ny, ny_hr)]
            else:
                psi_mtx = np.zeros([ny, ny_hr])
                for jj in range(0, ny):
                    for qq in range(0, ny_hr):
                        psi_mtx[jj, qq] = qd.lag_eval(yyb, jj, yyb_hr[qq])
                psi_mtxs[(ny, ny_hr)] = psi_mtx
                
            # Loop through cells to calculate error
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    # Cell information for quadrature
                    [th0, th1] = cell.pos[:]
                    dth = th1 - th0
                    [nth]  = cell.ndofs[:]
                    
                    [_, _, _, _, thb, _] = qd.quad_xyth(nnodes_th = nth)

                    if kwargs['ref_kind'] in ('ang', 'all'):
                        [nth_hr]  = [int(ang_res_offset + nth)]
                    else:
                        [nth_hr]  = [nth]
                    
                    [_, _, _, _, thb_hr, wth_hr] = qd.quad_xyth(nnodes_th = nth_hr)
                    
                    thf_hr = proj.push_forward(th0, th1, thb_hr).reshape([1, 1, nth_hr])
                    wth_hr = wth_hr.reshape([1, 1, nth_hr])
                    
                    if (nth, nth_hr) in xsi_mtxs.keys():
                        xsi_mtx = xsi_mtxs[(nth, nth_hr)]
                    else:
                        xsi_mtx = np.zeros([nth, nth_hr])
                        for aa in range(0, nth):
                            for rr in range(0, nth_hr):
                                xsi_mtx[aa, rr] = qd.lag_eval(thb, aa, thb_hr[rr])
                        xsi_mtxs[(nth, nth_hr)] = xsi_mtx
                        
                    # Calculate error
                    u_cell  = anl_sol(xxf_hr, yyf_hr, thf_hr)
                    uh_cell = num_sol.cols[col_key].cells[cell_key].vals
                    u_hr_cell = np.zeros([nx_hr, ny_hr, nth_hr])
                    for pp in range(0, nx_hr):
                        for qq in range(0, ny_hr):
                            for rr in range(0, nth_hr):
                                for ii in range(0, nx):
                                    phi_ip = phi_mtx[ii, pp]
                                    for jj in range(0, ny):
                                        psi_jq = psi_mtx[jj, qq]
                                        for aa in range(0, nth):
                                            xsi_ar = xsi_mtx[aa, rr]
                                            
                                            u_hr_cell[pp, qq, rr] += uh_cell[ii, jj, aa] * phi_ip * psi_jq * xsi_ar
                                            
                    local_err += (dx * dy * dth / 8.) * np.sum(wx_hr * wy_hr * wth_hr * (u_cell - u_hr_cell)**2)
                    
    err = MPI_comm.allreduce(local_err)
    err = np.sqrt(err/intg_u2[kwargs['key']])
    
    if kwargs['blocking']:
        MPI_comm.Barrier()
        
    return err
