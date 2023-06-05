from copy import deepcopy
import numpy as np
from time import perf_counter

from .Error_Indicator import Error_Indicator

import dg.quadrature as qd
from rt import rtdg

from utils import print_msg

phi_projs = {}
psi_projs = {}
xsi_projs = {}
spt_projs = {}

def high_res_err(mesh, proj, kappa, sigma, Phi, bcs_dirac, f, **kwargs):
    """
    Create a high-resolution version of the mesh, solve the problem there,
    project the solution from the low-resolution mesh onto the high-resolution
    mesh, and find max-norm error from there.
    """
    
    default_kwargs = {'solver' : 'spsolve',
                      'precondition' : False,
                      'verbose' : False}
    kwargs = {**default_kwargs, **kwargs}
    
    [bcs, dirac] = bcs_dirac
    
    mesh_hr = deepcopy(mesh)
    nref = 3
    for _ in range(0, nref):
        mesh_hr.ref_mesh(kind = 'all', form = 'p')

    proj_hr = rtdg(mesh_hr, kappa, sigma, Phi, [bcs, dirac], f, **kwargs)
    
    col_items = sorted(mesh.cols.items())
    ncols = len(col_items)
    
    err_ind = Error_Indicator(mesh, by_col = True, by_cell = True)

    # To get max-norm relative error, we need the maximal value of hr_proj
    max_proj_hr = 0.

    # Get max_norm(u_hr - uh) by column
    if kwargs['verbose']:
        t0 = perf_counter()
        
    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy] = [x1 - x0, y1 - y0]
            
            [nx, ny] = col.ndofs[:]
            [nx_hr, ny_hr] = mesh_hr.cols[col_key].ndofs[:]
            
            [xxb,    _,     yyb,    _,     _, _] = qd.quad_xyth(nnodes_x = nx,
                                                                nnodes_y = ny)
            [xxb_hr, wx_hr, yyb_hr, wy_hr, _, _] = qd.quad_xyth(nnodes_x = nx_hr,
                                                                nnodes_y = ny_hr)
            
            # Store spatial projection matrices for later reuse
            if (nx, nx_hr) in phi_projs.keys():
                phi_proj = phi_projs[(nx, nx_hr)][:]
            else:
                phi_proj = np.zeros([nx, nx_hr])
                for ii in range(0, nx):
                    for pp in range(0, nx_hr):
                        wx_pp = wx_hr[pp]
                        phi_proj[ii, pp] =  wx_pp * qd.lag_eval(xxb, ii, xxb_hr[pp])
                phi_projs[(nx, nx_hr)] = phi_proj[:]
                        
            if (ny, ny_hr) in psi_projs.keys():
                psi_proj = psi_projs[(ny, ny_hr)][:]
            else:
                psi_proj = np.zeros([ny, ny_hr])
                for jj in range(0, ny):
                    for qq in range(0, ny_hr):
                        wy_qq = wy_hr[qq]
                        psi_proj[jj, qq] = wy_qq * qd.lag_eval(yyb, jj, yyb_hr[qq])
                psi_projs[(ny, ny_hr)] = psi_proj[:]
                
            dcoeff_x = dx / 2.
            dcoeff_y = dy / 2.

            phi_proj = dcoeff_x * phi_proj[:]
            psi_proj = dcoeff_y * psi_proj[:]
            
            col_err = 0.
            
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, th1] = cell.pos[:]
                    dth = th1 - th0
                    
                    [nth]    = cell.ndofs[:]
                    [nth_hr] = mesh_hr.cols[col_key].cells[cell_key].ndofs[:]

                    [_, _, _, _, thb,    _     ] = qd.quad_xyth(nnodes_th = nth)
                    [_, _, _, _, thb_hr, wth_hr] = qd.quad_xyth(nnodes_th = nth_hr)
                    
                    uh_cell  = proj.cols[col_key].cells[cell_key].vals
                    uhr_cell = proj_hr.cols[col_key].cells[cell_key].vals
                    
                    uh_hr_cell = np.zeros_like(uhr_cell)
                    
                    # Store angular porjection matrices for later reuse                    
                    if (nth, nth_hr) in xsi_projs.keys():
                        xsi_proj = xsi_projs[(nth, nth_hr)][:]
                    else:
                        xsi_proj = np.zeros([nth, nth_hr])
                        for aa in range(0, nth):
                            for rr in range(0, nth_hr):
                                wth_rr = wth_hr[rr]
                                xsi_proj[aa, rr] = wth_rr * qd.lag_eval(thb, aa, thb_hr[rr])
                        xsi_projs[(nth, nth_hr)] = xsi_proj[:]
                        
                    xsi_proj = dth / 2. * xsi_proj[:]
                        
                    for pp in range(0, nx_hr):
                        for qq in range(0, ny_hr):
                            for rr in range(0, nth_hr):
                                for ii in range(0, nx):
                                    for jj in range(0, ny):
                                        for aa in range(0, nth):
                                            uh_hr_cell[pp, qq, rr] += uh_cell[ii, jj, aa] * phi_proj[ii, pp] * psi_proj[jj, qq] * xsi_proj[aa, rr]
                            
                    cell_err = np.amax(np.abs(uhr_cell - uh_hr_cell))
                    col_err = max(col_err, cell_err)
                    
                    
                    err_ind.cols[col_key].cells[cell_key].err_ind = cell_err
                    
                    max_proj_hr   = max(max_proj_hr, np.amax(np.abs(uhr_cell)))
            
            err_ind.cols[col_key].err_ind = col_err
            
    # Weight to be relative error
    for col_key, col in col_items:
        if col.is_lf:
            err_ind.cols[col_key].err_ind /= max_proj_hr

            err_ind.max_err = max(err_ind.max_err, err_ind.cols[col_key].err_ind)

            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    err_ind.cols[col_key].cells[cell_key].err_ind /= max_proj_hr

                    err_ind.max_err = max(err_ind.max_err,
                                          err_ind.cols[col_key].cells[cell_key].err_ind)

    if kwargs['verbose']:
        tf = perf_counter()
        msg = (
            'Hi-Res. Error Indicator Construction Time: {:8.4f} [s]\n'.format(tf - t0)
            )
        print_msg(msg)

    return err_ind
