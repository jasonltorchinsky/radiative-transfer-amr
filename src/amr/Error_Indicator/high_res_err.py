# Standard Library Imports
import copy
import os
import time

# Third-Party Library Imports
import numpy           as np
import petsc4py
from   mpi4py          import MPI
from   petsc4py        import PETSc

# Local Library Imports
import dg.mesh             as ji_mesh
import dg.projection       as proj
import dg.quadrature       as qd
import rt
import utils

from .Error_Indicator import Error_Indicator
from .hp_steer        import hp_steer_col, hp_steer_cell


# Library Variables
mesh_hr = None
uh_hr   = None
intg_uh_hr_2 = None

phi_projs = {}
psi_projs = {}
xsi_projs = {}

def high_res_err_new(mesh, uh_proj, kappa, sigma, Phi, bcs_dirac, f, **kwargs):
    """
    Construct a single high-resolution solution and save it to file for later
    reuse. Compare all numerical solutions against it.
    """
    
    default_kwargs = { "ndof_x_hr"  : 6, # x-DoFs for hi-res solution
                       "ndof_y_hr"  : 6, # y-DOFs for hi-res solution
                       "ndof_th_hr" : 6, # th-DoFs for hi-res solution
                       "nref_spt"   : 3, # Number of spatial h-refs
                       "nref_ang"   : 2, # Number of angular h-refs
                       "verbose"    : False, # Print info while executing
                       "blocking"   : True,  # Synchronize ranks before exiting
                       "file_path"  : "",    # Path for saving files
                       "uh_hr_file_name" : "uh_hr.npy" # File name for hr-soln
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
            [nx_hr, ny_hr, nth_hr] = [kwargs["ndof_x_hr"], kwargs["ndof_y_hr"],
                                      kwargs["ndof_th_hr"]]
            mesh_hr = ji_mesh.Mesh(Ls    = mesh.Ls[:],
                                   pbcs  = mesh.pbcs[:],
                                   ndofs = [nx_hr, ny_hr, nth_hr],
                                   has_th = mesh.has_th
                                   )
            for _ in range(0, kwargs["nref_ang"]):
                mesh_hr.ref_mesh(kind = "ang", form = "h")
            for _ in range(0, kwargs["nref_spt"]):
                mesh_hr.ref_mesh(kind = "spt", form = "h")
        else:
            mesh_hr = None
        mesh_hr = MPI_comm.bcast(mesh_hr, root = 0)
    mesh = MPI_comm.bcast(mesh, root = 0)
    uh_proj = MPI_comm.bcast(uh_proj, root = 0)
    
    # Get the high-resolution solution
    global uh_hr
    if uh_hr is None:
        uh_hr_file_path = os.path.join(kwargs["file_path"],
                                       kwargs["uh_hr_file_name"])
        if not os.path.isfile(uh_hr_file_path): # If not saved to file yet
            # Get the solution and save it to file
            [uh_hr, info] = rt.rtdg(mesh_hr, kappa, sigma, Phi, bcs_dirac, f,
                                    blocking = True,
                                    verbose = kwargs["verbose"])
            PETSc.garbage_cleanup()
            if comm_rank == 0:
                uh_hr_vec = uh_hr.to_vector()
                np.save(uh_hr_file_path, uh_hr_vec)
            MPI_comm.barrier()
            # Share the high-resolution solution across processes
            uh_hr = MPI_comm.bcast(uh_hr, root = 0)
            
            # If it"s not saved to file, assume we need to plot it
            if comm_rank == 0:
                if kwargs["verbose"]:
                    msg = (
                        "Plotting high-resolution solution...\n"
                    )
                    utils.print_msg(msg, blocking = False)
                    t0 = time.perf_counter()
                    
                # Plot high-resolution solution for posterity
                file_name = "uh_hr_th.png"
                file_path = os.path.join(kwargs["file_path"], file_name)
                proj.utils.plot_th(mesh_hr, uh_hr, file_name = file_path)
                
                file_name = "uh_hr_xy.png"
                file_path = os.path.join(kwargs["file_path"], file_name)
                proj.utils.plot_xy(mesh_hr, uh_hr, file_name = file_path)
                
                file_name = "uh_hr_xth.png"
                file_path = os.path.join(kwargs["file_path"], file_name)
                proj.utils.plot_xth(mesh_hr, uh_hr, file_name = file_path)
                
                file_name = "uh_hr_yth.png"
                file_path = os.path.join(kwargs["file_path"], file_name)
                proj.utils.plot_yth(mesh_hr, uh_hr, file_name = file_path)
                
                file_name = "uh_hr_xyth.png"
                file_path = os.path.join(kwargs["file_path"], file_name)
                proj.utils.plot_xyth(mesh_hr, uh_hr, file_name = file_path)
                
                if kwargs["verbose"]:
                    tf = time.perf_counter()
                    dt = tf - t0
                    msg = (
                        "High-resolution solution plotted!\n" +
                        12 * " " + "Time Elapsed: {:8.4f} [s]\n".format(dt)
                    )
                    utils.print_msg(msg, blocking = False)
        else: # If saved to file, read it from there
            uh_hr_vec = np.load(uh_hr_file_path)
            uh_hr     = proj.to_projection(mesh_hr, uh_hr_vec)
            
    # Integrate the square of the high-resolution solution for relative error
    global intg_uh_hr_2
    if intg_uh_hr_2 is None:
        if kwargs["verbose"]:
            msg = (
                "Integrating high-resolution solution...\n"
            )
            utils.print_msg(msg, blocking = False)
            t0 = time.perf_counter()
            
        # Split the problem into parts dependent on size of COMM_WORLD.
        col_keys_hr_global = list(sorted(mesh_hr.cols.keys()))
        col_keys_hr_local  = np.array_split(col_keys_hr_global, comm_size)[comm_rank].astype(np.int32)
        
        local_intg_uh_hr_2 = 0.
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
                        
                        uh_hr_cell = uh_hr.cols[col_key_hr].cells[cell_key_hr].vals[:, :, :]
                        
                        local_intg_uh_hr_2 += dcoeff * (dth / 2.) * np.sum(wx_hr * wy_hr * wth_hr * (uh_hr_cell)**2)
        intg_uh_hr_2 = MPI_comm.allreduce(local_intg_uh_hr_2)
        if kwargs["verbose"]:
            tf = time.perf_counter()
            dt = tf - t0
            msg = (
                "High-resolution solution integrated!\n" +
                12 * " " + "Time Elapsed: {:8.4f} [s]\n".format(dt)
            )
            utils.print_msg(msg, blocking = False)

    # Calculate the error
    # Embed the low-res soln in the hi-res mesh and get the L2-error
    if kwargs["verbose"]:
        msg = (
            "Calculating high-resolution error...\n"
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
                    uh_hr_cell = uh_hr.cols[col_key_hr].cells[cell_key_hr].vals[:,:,:]
                    uh_hr_cell_proj = np.zeros([nx_hr, ny_hr, nth_hr])
                    for pp_p in range(0, nx_hr):
                        for qq_p in range(0, ny_hr):
                            for rr_p in range(0, nth_hr):
                                for pp in range(0, nx_hr):
                                    phi_pp_p = phi_mtx_hr[pp, pp_p]
                                    for qq in range(0, ny_hr):
                                        psi_qq_p = psi_mtx_hr[qq, qq_p]
                                        for rr in range(0, nth_hr):
                                            xsi_rr_p = xsi_mtx_hr[rr, rr_p]
                                            
                                            uh_hr_cell_proj[pp_p, qq_p, rr_p] += \
                                                uh_hr_cell[pp, qq, rr] \
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
                                            
                    local_err += (dx * dy * dth / 8.) * np.sum(wx_hr * wy_hr * wth_hr * (uh_hr_cell_proj - uh_lr_cell_proj)**2)
    err = MPI_comm.allreduce(local_err)
    err = np.sqrt(err / intg_uh_hr_2)
    if kwargs["verbose"]:
        tf = time.perf_counter()
        dt = tf - t0
        msg = (
            "High-resolution error obtained!\n" +
            12 * " " + "Time Elapsed: {:8.4f} [s]\n".format(dt)
        )
        utils.print_msg(msg, blocking = False)
        
    if kwargs["blocking"]:
        MPI_comm.Barrier()
        
    return err

def high_res_err(mesh, uh_proj, kappa, sigma, Phi, bcs_dirac, f, **kwargs):
    """
    Refine the mesh twice in p, solve the problem, then calculate the error.
    """
    
    default_kwargs = {"ref_kind"   : "all",
                      "ang_res_offset" : 1,   # Factor to add to angular DoFs
                      "spt_res_offset" : 1,   # Factor to add to spatial DoFs
                      "verbose"    : False, # Print info while executing
                      "blocking"   : True, # Synchronize ranks before exiting
                      "out_dir"    : "out"
                      }
    kwargs = {**default_kwargs, **kwargs}
    
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    comm      = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()
    
    spt_res_offset = kwargs["spt_res_offset"]
    ang_res_offset = kwargs["ang_res_offset"]
    
    if comm_rank == 0:
        uh = uh_proj # Assign by reference
        
        # Get high-resolution mesh
        mesh_hr = copy.deepcopy(mesh)
        if kwargs["ref_kind"] in ("ang", "all"):
            for _ in range(0, ang_res_offset):
                mesh_hr.ref_mesh(kind = "ang", form = "p")
        if kwargs["ref_kind"] in ("spt", "all"):
            for _ in range(0, spt_res_offset):
                mesh_hr.ref_mesh(kind = "spt", form = "p")
            
        # Get high-resolution solution
        ndof_hr = mesh_hr.get_ndof()
        ndof_hr = MPI_comm.bcast(ndof_hr, root = 0)
    else:
        mesh_hr = None
        ndof_hr = None
        ndof_hr = MPI_comm.bcast(ndof_hr, root = 0)
    [uh_hr, info, _] = rt.rtdg(mesh_hr, kappa, sigma, Phi, bcs_dirac, f,
                            **kwargs)
    PETSc.garbage_cleanup()
    
    if comm_rank == 0:
        if kwargs["verbose"]:
            msg = (
                "Plotting high-resolution solution...\n"
            )
            utils.print_msg(msg, blocking = False)
            t0 = time.perf_counter()
            
        # Plot high-resolution solution for posterity
        #file_name = "uh_hr_th.png"
        #file_path = os.path.join(kwargs["dir_name"], file_name)
        #proj.utils.plot_th(mesh_hr, uh_hr, file_name = file_path)
        
        #file_name = "uh_hr_xy.png"
        #file_path = os.path.join(kwargs["dir_name"], file_name)
        #proj.utils.plot_xy(mesh_hr, uh_hr, file_name = file_path)
        
        #file_name = "uh_hr_xth.png"
        #file_path = os.path.join(kwargs["dir_name"], file_name)
        #proj.utils.plot_xth(mesh_hr, uh_hr, file_name = file_path)
        
        #file_name = "uh_hr_yth.png"
        #file_path = os.path.join(kwargs["dir_name"], file_name)
        #proj.utils.plot_yth(mesh_hr, uh_hr, file_name = file_path)
        
        #file_name = "uh_hr_xyth.png"
        #file_path = os.path.join(kwargs["dir_name"], file_name)
        #proj.utils.plot_xyth(mesh_hr, uh_hr, file_name = file_path)
        
        if kwargs["verbose"]:
            tf = time.perf_counter()
            dt = tf - t0
            msg = (
                "High-resolution solution plotted!\n" +
                12 * " " + "Time Elapsed: {:8.4f} [s]\n".format(dt)
            )
            utils.print_msg(msg, blocking = False)
            
        if kwargs["verbose"]:
            msg = (
                "Integrating high-resolution solution...\n"
            )
            utils.print_msg(msg, blocking = False)
            t0 = time.perf_counter()
            
        # Integrate high-resolution solution
        intg_uh_hr2   = 0.
        col_items_hr = sorted(mesh_hr.cols.items())
        for col_key_hr, col_hr in col_items_hr:
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
                        
                        uh_hr_cell = uh_hr.cols[col_key_hr].cells[cell_key_hr].vals[:, :, :]
                        
                        intg_uh_hr2 += dcoeff * (dth / 2.) * np.sum(wx_hr * wy_hr * wth_hr * (uh_hr_cell)**2)
                        
        if kwargs["verbose"]:
            tf = time.perf_counter()
            dt = tf - t0
            msg = (
                "High-resolution solution integrated!\n" +
                12 * " " + "Time Elapsed: {:8.4f} [s]\n".format(dt)
            )
            utils.print_msg(msg, blocking = False)
            
        if kwargs["verbose"]:
            msg = (
                "Calculating high-resolution error...\n"
            )
            utils.print_msg(msg, blocking = False)
            t0 = time.perf_counter()
            
        # Now calculate the error. We project high-resolution solution to low-resolution mesh.
        # This may get into some weird quadrature rule stuff, but that"s the bullet we"ll bite.
        # The meshs have the same column and cell keys.
        hi_res_err = 0.
        col_keys = sorted(mesh.cols.keys())
        for col_key in col_keys:
            col = mesh.cols[col_key]
            col_hr = mesh_hr.cols[col_key]
            if col.is_lf and col_hr.is_lf:
                [x0, y0, x1, y1] = col.pos[:]
                [dx, dy]         = [x1 - x0, y1 - y0]
                [nx, ny]         = col.ndofs[:]
                [nx_hr, ny_hr]   = col_hr.ndofs[:]
                
                [_, wx, _, wy, _, _] = qd.quad_xyth(nnodes_x = nx, nnodes_y = ny)
                wx  = wx.reshape([nx, 1, 1])
                wy  = wy.reshape([1, ny, 1])
                
                global phi_projs, psi_projs
                if (nx_hr, nx) in phi_projs.keys():
                    phi_mtx = phi_projs[(nx_hr, nx)][:,:]
                else:
                    [xxb,    _, _, _, _, _] = qd.quad_xyth(nnodes_x = nx)
                    [xxb_hr, _, _, _, _, _] = qd.quad_xyth(nnodes_x = nx_hr)
                    
                    phi_mtx = np.zeros([nx_hr, nx])
                    for pp in range(0, nx_hr):
                        for ii in range(0, nx):
                            phi_mtx[pp, ii] = qd.lag_eval(xxb_hr, pp, xxb[ii])
                    phi_projs[(nx_hr, nx)] = phi_mtx[:,:]
                    
                if (ny_hr, ny) in psi_projs.keys():
                    psi_mtx = psi_projs[(ny_hr, ny)][:,:]
                else:
                    [_, _, yyb,    _, _, _] = qd.quad_xyth(nnodes_y = ny)
                    [_, _, yyb_hr, _, _, _] = qd.quad_xyth(nnodes_y = ny_hr)
                    
                    psi_mtx = np.zeros([ny_hr, ny])
                    for qq in range(0, ny_hr):
                        for jj in range(0, ny):
                            psi_mtx[qq, jj] = qd.lag_eval(yyb_hr, qq, yyb[jj])
                    psi_projs[(ny_hr, ny)] = psi_mtx[:,:]
                    
                # The cols have the same cell keys
                cell_keys = sorted(col.cells.keys())
                
                for cell_key in cell_keys:
                    cell = col.cells[cell_key]
                    cell_hr = col_hr.cells[cell_key]
                    if cell.is_lf and cell_hr.is_lf:
                        [th0, th1] = cell.pos[:]
                        dth        = th1 - th0
                        [nth]      = cell.ndofs[:]
                        [nth_hr]   = cell_hr.ndofs[:]
                        
                        [_, _, _, _, _, wth] = qd.quad_xyth(nnodes_th = nth)
                        wth = wth.reshape([1, 1, nth])

                        global xsi_projs
                        if (nth_hr, nth) in xsi_projs.keys():
                            xsi_mtx = xsi_projs[(nth_hr, nth)][:,:]
                        else:
                            [_, _, _, _, thb,    _] = qd.quad_xyth(nnodes_th = nth)
                            [_, _, _, _, thb_hr, _] = qd.quad_xyth(nnodes_th = nth_hr)
                            
                            xsi_mtx = np.zeros([nth_hr, nth])
                            for rr in range(0, nth_hr):
                                for aa in range(0, nth):
                                    xsi_mtx[rr, aa] = qd.lag_eval(thb_hr, rr, thb[aa])
                            xsi_projs[(nth_hr, nth)] = xsi_mtx[:,:]
                            
                        uh_lr_cell = uh.cols[col_key].cells[cell_key].vals[:, :, :]
                        
                        uh_hr_cell = uh_hr.cols[col_key].cells[cell_key].vals[:, :, :]
                        
                        # Project high-resolution solution onto low-resolution mesh
                        uh_hr_lr_cell = np.zeros([nx, ny, nth])
                        for ii in range(0, nx):
                            for jj in range(0, ny):
                                for aa in range(0, nth):
                                    for pp in range(0, nx_hr):
                                        phi_pi = phi_mtx[pp, ii]
                                        for qq in range(0, ny_hr):
                                            psi_qj = psi_mtx[qq, jj]
                                            for rr in range(0, nth_hr):
                                                xsi_ra = xsi_mtx[rr, aa]
                                                
                                                uh_hr_lr_cell[ii, jj, aa] += uh_hr_cell[pp, qq, rr] * phi_pi * psi_qj * xsi_ra
                                                
                        hi_res_err += (dx * dy * dth / 8.) * np.sum(wx * wy * wth * (uh_hr_lr_cell - uh_lr_cell)**2)
                        
        err = np.sqrt(hi_res_err / intg_uh_hr2)
        if kwargs["verbose"]:
            tf = time.perf_counter()
            dt = tf - t0
            msg = (
                "High-resolution error obtained!\n" +
                12 * " " + "Time Elapsed: {:8.4f} [s]\n".format(dt)
            )
            utils.print_msg(msg, blocking = False)
    else:
        err = 0.
        
    err = MPI_comm.bcast(err, root = 0)
    
    if kwargs["blocking"]:
        MPI_comm.Barrier()
        
    return err
