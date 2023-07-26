import copy
import numpy           as np
import os
import petsc4py
import scipy.integrate as integrate
import time
from   mpi4py          import MPI
from   petsc4py        import PETSc

import dg.mesh             as ji_mesh
import dg.projection       as proj
import dg.projection.utils
import dg.quadrature       as qd
import rt
import utils

from .Error_Indicator import Error_Indicator
from .hp_steer        import hp_steer_col, hp_steer_cell

phi_projs = {}
psi_projs = {}
xsi_projs = {}

def high_res_err(mesh, uh_proj, kappa, sigma, Phi, bcs_dirac, f, **kwargs):
    """
    Refine the mesh twice in p, solve the problem, then calculate the error.
    """

    default_kwargs = {'ref_kind'  : 'all',
                      'verbose'   : False, # Print info while executing
                      'blocking'  : True # Synchronize ranks before exiting
                      }
    kwargs = {**default_kwargs, **kwargs}


    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    comm      = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()
    
    if comm_rank == 0:
        uh = uh_proj # Assign by reference
        
        # Get high-resolution mesh
        mesh_hr = copy.deepcopy(mesh)
        if kwargs['ref_kind'] in ('ang', 'all'):
            for _ in range(0, 2):
                mesh_hr.ref_mesh(kind = 'ang', form = 'p')
        if kwargs['ref_kind'] in ('spt', 'all'):
            for _ in range(0, 1):
                mesh_hr.ref_mesh(kind = 'spt', form = 'p')
            
        # Get high-resolution solution
        ndof_hr = mesh_hr.get_ndof()
        ndof_hr = MPI_comm.bcast(ndof_hr, root = 0)
    else:
        mesh_hr = None
        ndof_hr = None
        ndof_hr = MPI_comm.bcast(ndof_hr, root = 0)
    [uh_hr, info] = rt.rtdg(mesh_hr, kappa, sigma, Phi, bcs_dirac, f,
                            blocking = False, verbose = kwargs['verbose'])
    PETSc.garbage_cleanup()
    
    if comm_rank == 0:
        if kwargs['verbose']:
            msg = (
                'Plotting high-resolution solution...\n'
            )
            utils.print_msg(msg, blocking = False)
            t0 = time.perf_counter()
            
        # Plot high-resolution solution for posterity
        file_name = 'uh_hr_th.png'
        file_path = os.path.join(kwargs['dir_name'], file_name)
        proj.utils.plot_th(mesh_hr, uh_hr, file_name = file_path)
        
        file_name = 'uh_hr_xy.png'
        file_path = os.path.join(kwargs['dir_name'], file_name)
        proj.utils.plot_xy(mesh_hr, uh_hr, file_name = file_path)
        
        file_name = 'uh_hr_xth.png'
        file_path = os.path.join(kwargs['dir_name'], file_name)
        proj.utils.plot_xth(mesh_hr, uh_hr, file_name = file_path)
        
        file_name = 'uh_hr_yth.png'
        file_path = os.path.join(kwargs['dir_name'], file_name)
        proj.utils.plot_yth(mesh_hr, uh_hr, file_name = file_path)
        
        file_name = 'uh_hr_xyth.png'
        file_path = os.path.join(kwargs['dir_name'], file_name)
        proj.utils.plot_xyth(mesh_hr, uh_hr, file_name = file_path)
        
        if kwargs['verbose']:
            tf = time.perf_counter()
            dt = tf - t0
            msg = (
                'High-resolution solution plotted!\n' +
                12 * ' ' + 'Time Elapsed: {:8.4f} [s]\n'.format(dt)
            )
            utils.print_msg(msg, blocking = False)
            
        if kwargs['verbose']:
            msg = (
                'Integrating  high-resolution solution...\n'
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
                        
        if kwargs['verbose']:
            tf = time.perf_counter()
            dt = tf - t0
            msg = (
                'High-resolution solution integrated!\n' +
                12 * ' ' + 'Time Elapsed: {:8.4f} [s]\n'.format(dt)
            )
            utils.print_msg(msg, blocking = False)
            
        if kwargs['verbose']:
            msg = (
                'Calculating high-resolution error...\n'
            )
            utils.print_msg(msg, blocking = False)
            t0 = time.perf_counter()
            
        # Now calculate the error. We project high-resolution solution to low-resolution mesh.
        # This may get into some weird quadrature rule stuff, but that's the bullet we'll bite.
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
        if kwargs['verbose']:
            tf = time.perf_counter()
            dt = tf - t0
            msg = (
                'High-resolution error obtained!\n' +
                12 * ' ' + 'Time Elapsed: {:8.4f} [s]\n'.format(dt)
            )
            utils.print_msg(msg, blocking = False)
    else:
        err = 0.
        
    err = MPI_comm.bcast(err, root = 0)
    
    if kwargs['blocking']:
        MPI_comm.Barrier()
        
    return err
