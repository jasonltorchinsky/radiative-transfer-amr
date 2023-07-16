from copy import deepcopy
import numpy as np
from time import perf_counter
import os

from .Error_Indicator import Error_Indicator
from .hp_steer import hp_steer_col, hp_steer_cell

from dg.mesh import Mesh, calc_col_key, calc_cell_key
from dg.projection import Projection, to_projection, push_forward, pull_back
from dg.projection.utils import plot_xy, plot_xth, plot_yth, plot_xyth, plot_th
import dg.quadrature as qd
from rt import rtdg

from utils import print_msg

phi_projs = {}
psi_projs = {}
xsi_projs = {}

mesh_hr     = None
uh_hr       = None
intg_uh_hr2 = None

def high_res_err(mesh, proj, kappa, sigma, Phi, bcs_dirac, f, **kwargs):
    return high_res_err_lr(mesh, proj, kappa, sigma, Phi, bcs_dirac, f, **kwargs)

def high_res_err_hr(mesh, proj, kappa, sigma, Phi, bcs_dirac, f, **kwargs):
    """
    Create a high-resolution mesh of known refinement, calculate the solution
    on that mesh and save it to file for reuse.
    
    Don't bother calculating the error indicator, just the error.
    """

    default_kwargs = {'dir_name'        : None,
                      'uh_hr_file_name' : 'uh_hr.dat',
                      'nref_ang' : 3,
                      'nref_spt' : 2,
                      'ndof_x' : 8,
                      'ndof_y' : 8,
                      'ndof_th' : 16}
    kwargs = {**default_kwargs, **kwargs}


    nx_hr  = kwargs['ndof_x']
    ny_hr  = kwargs['ndof_y']
    nth_hr = kwargs['ndof_th']
    
    uh = proj # Assign by reference
    
    global mesh_hr, uh_hr, intg_uh_hr2
    
    # Get high-resolution mesh
    if mesh_hr is None:
        mesh_hr = Mesh(Ls = mesh.Ls[:],
                       pbcs = mesh.pbcs[:],
                       ndofs = [nx_hr, ny_hr, nth_hr],
                       has_th = True)
        for _ in range(0, kwargs['nref_ang']):
            mesh_hr.ref_mesh(kind = 'ang', form = 'h')
        for _ in range(0, kwargs['nref_spt']):
            mesh_hr.ref_mesh(kind = 'spt', form = 'h')
    else:
        mesh_hr = mesh_hr
        
    # Get high-resolution solution
    if uh_hr is None:
        uh_hr_file_path = os.path.join(kwargs['dir_name'],  kwargs['uh_hr_file_name'])
        if os.path.isfile(uh_hr_file_path): # We put the high-resolution solution in a file.
            uh_hr_file_path = os.path.join(kwargs['dir_name'], kwargs['uh_hr_file_name'])
            uh_hr_vec = np.fromfile(uh_hr_file_path)
            uh_hr = to_projection(mesh_hr, uh_hr_vec)
        else:
            [bcs, dirac] = bcs_dirac
            ndof_hr = mesh_hr.get_ndof()
            if ndof_hr <= 2**17:
                [uh_hr, info] = rtdg(mesh_hr, kappa, sigma, Phi, [bcs, dirac], f,
                                     solver = 'spsolve', **kwargs)
            else:
                [uh_hr, info] = rtdg(mesh_hr, kappa, sigma, Phi, [bcs, dirac], f,
                                     **kwargs)
            uh_hr_vec = uh_hr.to_vector()
            uh_hr_file_path = os.path.join(kwargs['dir_name'], 'uh_hr.dat')
            uh_hr_vec.tofile(uh_hr_file_path)
            
        # Plot high-resolution solution for posterity
        file_name = 'uh_hr_th.png'
        file_path = os.path.join(kwargs['dir_name'], file_name)
        plot_th(mesh_hr, uh_hr, file_name = file_path)
        
        file_name = 'uh_hr_xy.png'
        file_path = os.path.join(kwargs['dir_name'], file_name)
        plot_xy(mesh_hr, uh_hr, file_name = file_path)
        
        file_name = 'uh_hr_xth.png'
        file_path = os.path.join(kwargs['dir_name'], file_name)
        plot_xth(mesh_hr, uh_hr, file_name = file_path)
        
        file_name = 'uh_hr_yth.png'
        file_path = os.path.join(kwargs['dir_name'], file_name)
        plot_yth(mesh_hr, uh_hr, file_name = file_path)
        
        file_name = 'uh_hr_xyth.png'
        file_path = os.path.join(kwargs['dir_name'], file_name)
        plot_xyth(mesh_hr, uh_hr, file_name = file_path)

    # Integrate high-resolution solution
    if intg_uh_hr2 is None:
        intg_uh_hr2   = 0.
        col_items_hr = sorted(mesh_hr.cols.items())
        for col_key_hr, col_hr in col_items_hr:
            if col_hr.is_lf:
                [x0, y0, x1, y1] = col_hr.pos[:]
                [dx, dy] = [x1 - x0, y1 - y0]
                [_, wx_hr, _, wy_hr, _, _] = qd.quad_xyth(nnodes_x = nx_hr,
                                                          nnodes_y = ny_hr)
                wx_hr = wx_hr.reshape([nx_hr, 1, 1])
                wy_hr = wy_hr.reshape([1, ny_hr, 1])
                
                dcoeff = (dx * dy) / 4.
                
                cell_items_hr = sorted(col_hr.cells.items())
                for cell_key_hr, cell_hr in cell_items_hr:
                    if cell_hr.is_lf:
                        [th0, th1] = cell_hr.pos[:]
                        dth        = th1 - th0
                        
                        [_, _, _, _, _, wth_hr] = qd.quad_xyth(nnodes_th = nth_hr)
                        
                        wth_hr = wth_hr.reshape([1, 1, nth_hr])
                        
                        uh_hr_cell = uh_hr.cols[col_key_hr].cells[cell_key_hr].vals[:, :, :]
                        
                        intg_uh_hr2 += dcoeff * (dth / 2.) * np.sum(wx_hr * wy_hr * wth_hr * (uh_hr_cell)**2)
                        
    # Now calculate the error. We project high-resolution solution to low-resolution mesh.
    # This may get into some weird quadrature rule stuff, but that's the bullet we'll bite.
    hi_res_err = 0.
    col_items = sorted(mesh.cols.items())
    col_items_hr = sorted(mesh_hr.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy]         = [x1 - x0, y1 - y0]
            [xmid, ymid]     = [(x1 + x0) / 2., (y1 + y0) / 2.]
            [nx, ny]         = col.ndofs[:]
            
            [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = nx,
                                                    nnodes_y = ny)
            xxf = push_forward(x0, x1, xxb)
            yyf = push_forward(y0, y1, yyb)
            wx = wx.reshape([nx, 1, 1])
            wy = wy.reshape([1, ny, 1])
            
            cell_items = sorted(col.cells.items())
            
            for col_key_hr, col_hr in col_items_hr:
                if col_hr.is_lf:
                    [x0_hr, y0_hr, x1_hr, y1_hr] = col_hr.pos[:]
                    
                    if ((x0_hr <= xmid) and (xmid <= x1_hr)) and ((y0_hr <= ymid) and (ymid < y1_hr)):
                        [xxb_hr, _, yyb_hr, _, _, _] = qd.quad_xyth(nnodes_x = nx_hr,
                                                                    nnodes_y = ny_hr)
                        
                        xxb_lr_hr = pull_back(x0_hr, x1_hr, xxf)
                        yyb_lr_hr = pull_back(y0_hr, y1_hr, yyf)
                        
                        phi_mtx = np.zeros([nx_hr, nx])
                        for pp in range(0, nx_hr):
                            for ii in range(0, nx):
                                phi_mtx[pp, ii] = qd.lag_eval(xxb_hr, pp, xxb_lr_hr[ii])
                                
                        psi_mtx = np.zeros([ny_hr, ny])
                        for qq in range(0, ny_hr):
                            for jj in range(0, ny):
                                psi_mtx[qq, jj] = qd.lag_eval(yyb_hr, qq, yyb_lr_hr[jj])
                        
                        cell_items_hr = sorted(col_hr.cells.items())
                        
                        for cell_key, cell in cell_items:
                            if cell.is_lf:
                                [th0, th1] = cell.pos[:]
                                dth = th1 - th0
                                thmid = (th1 + th0) / 2.
                                [nth] = cell.ndofs[:]
                                
                                [_, _, _, _, thb, wth] = qd.quad_xyth(nnodes_th = nth)
                                thf = push_forward(th0, th1, thb)
                                wth = wth.reshape([1, 1, nth])
                                
                                uh_lr_cell = uh.cols[col_key].cells[cell_key].vals[:, :, :]
                                
                                for cell_key_hr, cell_hr in cell_items_hr:
                                    [th0_hr, th1_hr] = cell_hr.pos[:]
                                    
                                    if ((th0_hr <= thmid) and (thmid <= th1_hr)):
                                        [_, _, _, _, thb_hr, _] = qd.quad_xyth(nnodes_th = nth_hr)
                                        
                                        thb_lr_hr = pull_back(th0, th1, thf)
                                        
                                        xsi_mtx = np.zeros([nth_hr, nth])
                                        for rr in range(0, nth_hr):
                                            for aa in range(0, nth):
                                                xsi_mtx[rr, aa] = qd.lag_eval(thb_hr, rr, thb_lr_hr[aa])
                                                
                                        uh_hr_cell = uh_hr.cols[col_key_hr].cells[cell_key_hr].vals[:, :, :]
                                        
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
                                        
    return np.sqrt(hi_res_err / intg_uh_hr2)

def high_res_err_lr(mesh, proj, kappa, sigma, Phi, bcs_dirac, f, **kwargs):
    """
    Refine the mesh twice in p, solve the problem, then calculate the error.
    """

    default_kwargs = {'ref_kind' : 'all'}
    kwargs = {**default_kwargs, **kwargs}
    
    uh = proj # Assign by reference
    
    # Get high-resolution mesh
    mesh_hr = deepcopy(mesh)
    for _ in range(0, 2):
        mesh_hr.ref_mesh(kind = kwargs['ref_kind'], form = 'p')
        
    # Get high-resolution solution
    ndof_hr = mesh_hr.get_ndof()
    if ndof_hr <= 2**17:
        [uh_hr, info] = rtdg(mesh_hr, kappa, sigma, Phi, bcs_dirac, f,
                             solver = 'spsolve', **kwargs)
    else:
        [uh_hr, info] = rtdg(mesh_hr, kappa, sigma, Phi, bcs_dirac, f,
                             solver = 'gmres', precondition = True, **kwargs)
        
    # Plot high-resolution solution for posterity
    file_name = 'uh_hr_th.png'
    file_path = os.path.join(kwargs['dir_name'], file_name)
    plot_th(mesh_hr, uh_hr, file_name = file_path)
    
    file_name = 'uh_hr_xy.png'
    file_path = os.path.join(kwargs['dir_name'], file_name)
    plot_xy(mesh_hr, uh_hr, file_name = file_path)
    
    file_name = 'uh_hr_xth.png'
    file_path = os.path.join(kwargs['dir_name'], file_name)
    plot_xth(mesh_hr, uh_hr, file_name = file_path)
    
    file_name = 'uh_hr_yth.png'
    file_path = os.path.join(kwargs['dir_name'], file_name)
    plot_yth(mesh_hr, uh_hr, file_name = file_path)
    
    file_name = 'uh_hr_xyth.png'
    file_path = os.path.join(kwargs['dir_name'], file_name)
    plot_xyth(mesh_hr, uh_hr, file_name = file_path)

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
                        
    # Now calculate the error. We project high-resolution solution to low-resolution mesh.
    # This may get into some weird quadrature rule stuff, but that's the bullet we'll bite.
    hi_res_err = 0.
    col_items = sorted(mesh.cols.items())
    col_items_hr = sorted(mesh_hr.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy]         = [x1 - x0, y1 - y0]
            [xmid, ymid]     = [(x1 + x0) / 2., (y1 + y0) / 2.]
            [nx, ny]         = col.ndofs[:]
            
            [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = nx,
                                                    nnodes_y = ny)
            xxf = push_forward(x0, x1, xxb)
            yyf = push_forward(y0, y1, yyb)
            wx = wx.reshape([nx, 1, 1])
            wy = wy.reshape([1, ny, 1])
            
            cell_items = sorted(col.cells.items())
            
            for col_key_hr, col_hr in col_items_hr:
                if col_hr.is_lf:
                    [nx_hr, ny_hr]   = col_hr.ndofs[:]
                    [x0_hr, y0_hr, x1_hr, y1_hr] = col_hr.pos[:]
                    
                    if ((x0_hr <= xmid) and (xmid <= x1_hr)) and ((y0_hr <= ymid) and (ymid < y1_hr)):
                        [xxb_hr, _, yyb_hr, _, _, _] = qd.quad_xyth(nnodes_x = nx_hr,
                                                                    nnodes_y = ny_hr)
                        
                        xxb_lr_hr = pull_back(x0_hr, x1_hr, xxf)
                        yyb_lr_hr = pull_back(y0_hr, y1_hr, yyf)
                        
                        phi_mtx = np.zeros([nx_hr, nx])
                        for pp in range(0, nx_hr):
                            for ii in range(0, nx):
                                phi_mtx[pp, ii] = qd.lag_eval(xxb_hr, pp, xxb_lr_hr[ii])
                                
                        psi_mtx = np.zeros([ny_hr, ny])
                        for qq in range(0, ny_hr):
                            for jj in range(0, ny):
                                psi_mtx[qq, jj] = qd.lag_eval(yyb_hr, qq, yyb_lr_hr[jj])
                        
                        cell_items_hr = sorted(col_hr.cells.items())
                        
                        for cell_key, cell in cell_items:
                            if cell.is_lf:
                                [th0, th1] = cell.pos[:]
                                dth   = th1 - th0
                                thmid = (th1 + th0) / 2.
                                [nth] = cell.ndofs[:]
                                
                                [_, _, _, _, thb, wth] = qd.quad_xyth(nnodes_th = nth)
                                thf = push_forward(th0, th1, thb)
                                wth = wth.reshape([1, 1, nth])
                                
                                uh_lr_cell = uh.cols[col_key].cells[cell_key].vals[:, :, :]
                                
                                for cell_key_hr, cell_hr in cell_items_hr:
                                    [nth_hr]         = cell_hr.ndofs[:]
                                    [th0_hr, th1_hr] = cell_hr.pos[:]
                                    
                                    if ((th0_hr <= thmid) and (thmid <= th1_hr)):
                                        [_, _, _, _, thb_hr, _] = qd.quad_xyth(nnodes_th = nth_hr)
                                        
                                        thb_lr_hr = pull_back(th0, th1, thf)
                                        
                                        xsi_mtx = np.zeros([nth_hr, nth])
                                        for rr in range(0, nth_hr):
                                            for aa in range(0, nth):
                                                xsi_mtx[rr, aa] = qd.lag_eval(thb_hr, rr, thb_lr_hr[aa])
                                                
                                        uh_hr_cell = uh_hr.cols[col_key_hr].cells[cell_key_hr].vals[:, :, :]
                                        
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
                                        
    return np.sqrt(hi_res_err / intg_uh_hr2)
