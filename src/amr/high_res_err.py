from copy import deepcopy
import numpy as np

from .Error_Indicator import Error_Indicator

import dg.quadrature as qd
from rt import rtdg

def high_res_err(mesh, proj, kappa, sigma, Phi, bcs_dirac, f):
    """
    Create a high-resolution version of the mesh, solve the problem there,
    project the solution from the low-resolution mesh onto the high-resolution
    mesh, and find max-norm error from there.
    """
    
    [bcs, dirac] = bcs_dirac
    
    mesh_hr = deepcopy(mesh)
    nref = 3
    for _ in range(0, nref):
        mesh_hr.ref_mesh(kind = 'all', form = 'p')

    proj_hr = rtdg(mesh_hr, kappa, sigma, Phi, [bcs, dirac], f)
    
    col_items = sorted(mesh.cols.items())
    ncols = len(col_items)
    
    err_ind = Error_Indicator(mesh, by_col = True, by_cell = True)

    # To get max-norm relative error, we need the maximal value of hr_proj
    max_proj_hr = 0.

    # Get max_norm(u_hr - uh) by column
    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy] = [x1 - x0, y1 - y0]
            
            [nx, ny] = col.ndofs[:]
            [nx_hr, ny_hr] = [nx + nref, ny + nref]

            [xxb,    _,     yyb,    _,     _, _] = qd.quad_xyth(nnodes_x = nx,
                                                                nnodes_y = ny)
            [xxb_hr, wx_hr, yyb_hr, wy_hr, _, _] = qd.quad_xyth(nnodes_x = nx_hr,
                                                                nnodes_y = ny_hr)
            
            col_err = 0.
            
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, th1] = cell.pos[:]
                    dth = th1 - th0
                    
                    [nth] = cell.ndofs[:]
                    nth_hr = nth + nref

                    [_, _, _, _, thb,    _     ] = qd.quad_xyth(nnodes_th = nth)
                    [_, _, _, _, thb_hr, wth_hr] = qd.quad_xyth(nnodes_th = nth_hr)
                    
                    uh_cell  = proj.cols[col_key].cells[cell_key].vals
                    uhr_cell = proj_hr.cols[col_key].cells[cell_key].vals

                    uh_hr_cell = np.zeros_like(uhr_cell)
                    dcoeff = dx * dy * dth / 8
                    for pp in range(0, nx_hr):
                        wx_p = wx_hr[pp]
                        for qq in range(0, ny_hr):
                            wy_q = wy_hr[qq]
                            for rr in range(0, nth_hr):
                                wth_r = wth_hr[rr]
                                for ii in range(0, nx):
                                    phi_ip = qd.lag_eval(xxb, ii, xxb_hr[pp])
                                    for jj in range(0, ny):
                                        psi_jq = qd.lag_eval(yyb, jj, yyb_hr[qq])
                                        for aa in range(0, nth):
                                            xsi_ar = qd.lag_eval(thb, aa, thb_hr[rr])

                                            uh_hr_cell[pp, qq, rr] += dcoeff \
                                                * wx_p * wy_q * wth_r \
                                                * phi_ip * psi_jq * xsi_ar

                                
                    
                    col_err = max(col_err, np.amax(np.abs(uhr_cell - uh_hr_cell)))
                    cell_err = np.amax(np.abs(uhr_cell - uh_hr_cell))
                    
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

                    

    return err_ind
