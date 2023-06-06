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


def high_res_err_old(mesh, proj, kappa, sigma, Phi, bcs_dirac, f, **kwargs):
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

    # X2 the polynomial degree in each element, h- refine everything once
    mult = 2
    mesh_hr = deepcopy(mesh)
    col_hr_items = sorted(mesh_hr.cols.items())
    for col_hr_key, col_hr in col_hr_items:
        if col_hr.is_lf:
            col_hr.ndofs[0] *= mult
            col_hr.ndofs[1] *= mult

            cell_hr_items = sorted(col_hr.cells.items())
            for cell_hr_key, cell_hr in cell_hr_items:
                if cell_hr.is_lf:
                    cell_hr.ndofs[0] *= mult

    mesh_hr.ref_mesh(kind = 'all', form = 'h')

    proj_hr = rtdg(mesh_hr, kappa, sigma, Phi, [bcs, dirac], f, **kwargs)
    
    err_ind = Error_Indicator(mesh, by_col = True, by_cell = True)

    # To get max-norm relative error, we need the maximal value of hr_proj
    max_proj_hr = 0.

    # Get max_norm(u_hr - uh) by column
    if kwargs['verbose']:
        t0 = perf_counter()
        
    for col_hr_key, col_hr in col_hr_items:
        if col_hr.is_lf:
            # Get col in mesh that col_hr is contained in
            # ASSUMES ONE h_REFINEMENT
            [col_hr_i, col_hr_j] = col_hr.idx[:]
            col_hr_lv = col_hr.lv
            
            col_idx = [int(col_hr_i/2), int(col_hr_j/2)]
            # Get which child col_hr is of col
            if int(col_hr_i/2) == col_hr_i:
                i_str = '-'
            else:
                i_str = '+'

            if int(col_hr_j/2) == col_hr_j:
                j_str = '-'
            else:
                j_str = '+'
            col_chld_str = i_str + j_str
            
            col_lv  = int(col_hr_lv - 1)
            col_key = calc_col_key(col_idx, col_lv)
            # Assume this is a leaf
            col     = mesh.cols[col_key]
            
            # Store spatial projection matrices for later reuse
            if (nx, nx_hr, col_chld_str) in phi_projs.keys():
                phi_proj = phi_projs[(nx, nx_hr, col_chld_str)][:]
            else:
                [nx, _] = col.ndofs[:]
                [x0, _, x1, _] = col.pos[:]
                [xxb, _, _, _, _, _] = qd.quad_xyth(nnodes_x = nx)
                
                [nx_hr, _] = col_hr.ndofs[:]
                [x0_hr, _, x1_hr, _] = col_hr.pos[:]
                [xxb_hr, _, _, _, _, _] = qd.quad_xyth(nnodes_x = nx_hr)
                
                # Must push-forward, pull back for proper coordinate evaluation
                xxf_hr = push_forward(x0_hr, x1_hr, xxb_hr)
                xxb_hr = pull_back(x0, x1, xxf_hr)
                
                phi_proj = np.zeros([nx, nx_hr])
                for ii in range(0, nx):
                    for pp in range(0, nx_hr):
                        phi_proj[ii, pp] =  qd.lag_eval(xxb, ii, xxb_hr[pp])
                phi_projs[(nx, nx_hr, col_chld_str)] = phi_proj[:]
                
            if (ny, ny_hr, col_chld_str) in psi_projs.keys():
                psi_proj = psi_projs[(ny, ny_hr, col_chld_str)][:]
            else:
                [_, ny] = col.ndofs[:]
                [_, y0, _, y1] = col.pos[:]
                [_, _, yyb, __, _, _] = qd.quad_xyth(nnodes_y = ny)
                
                [_, ny_hr] = col_hr.ndofs[:]
                [_, y0_hr, _, y1_hr] = col_hr.pos[:]
                [_, _, yyb_hr, _, _, _] = qd.quad_xyth(nnodes_y = ny_hr)
                
                # Must push-forward, pull back for proper coordinate evaluation
                yyf_hr = push_forward(y0_hr, y1_hr, yyb_hr)
                yyb_hr = pull_back(y0, y1, yyf_hr)
                
                psi_proj = np.zeros([ny, ny_hr])
                for jj in range(0, ny):
                    for qq in range(0, ny_hr):
                        psi_proj[jj, qq] = qd.lag_eval(yyb, jj, yyb_hr[qq])
                psi_projs[(ny, ny_hr, col_chld_str)] = psi_proj[:]
                
            col_err = 0.
            
            cell_hr_items = sorted(col_hr.cells.items())
            for cell_hr_key, cell_hr in cell_hr_items:
                if cell_hr.is_lf:
                    # Get cell in mesh that cell_hr is contained in
                    # ASSUMES ONE h_REFINEMENT
                    ## CONTINUE FROM HERE
                    [th0, th1] = cell.pos[:]
                    
                    [nth]    = cell.ndofs[:]
                    [nth_hr] = mesh_hr.cols[col_key].cells[cell_key].ndofs[:]

                    [_, _, _, _, thb,    _] = qd.quad_xyth(nnodes_th = nth)
                    [_, _, _, _, thb_hr, _] = qd.quad_xyth(nnodes_th = nth_hr)
                    
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
                                xsi_proj[aa, rr] = qd.lag_eval(thb, aa, thb_hr[rr])
                        xsi_projs[(nth, nth_hr)] = xsi_proj[:]
                        
                    for pp in range(0, nx_hr):
                        for qq in range(0, ny_hr):
                            for rr in range(0, nth_hr):
                                for ii in range(0, nx):
                                    phi_ip = phi_proj[ii, pp]
                                    for jj in range(0, ny):
                                        psi_jq = psi_proj[jj, qq]
                                        for aa in range(0, nth):
                                            xsi_ar = xsi_proj[aa, rr]
                                            uh_hr_cell[pp, qq, rr] += uh_cell[ii, jj, aa] \
                                                * phi_ip * psi_jq * xsi_ar
                            
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


def high_res_err_old(mesh, proj, kappa, sigma, Phi, bcs_dirac, f, **kwargs):
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

    # X2 the polynomial degree in each element, h- refine everything once
    mult = 2
    mesh_hr = deepcopy(mesh)
    col_hr_items = sorted(mesh_hr.cols.items())
    for col_hr_key, col_hr in col_hr_items:
        if col_hr.is_lf:
            col_hr.ndofs[0] *= mult
            col_hr.ndofs[1] *= mult

            cell_hr_items = sorted(col_hr.cells.items())
            for cell_hr_key, cell_hr in cell_hr_items:
                if cell_hr.is_lf:
                    cell_hr.ndofs[0] *= mult

    proj_hr = rtdg(mesh_hr, kappa, sigma, Phi, [bcs, dirac], f, **kwargs)
    
    col_items = sorted(mesh.cols.items())
    
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
            
            [xxb,    _, yyb,    _, _, _] = qd.quad_xyth(nnodes_x = nx,
                                                        nnodes_y = ny)
            [xxb_hr, _, yyb_hr, _, _, _] = qd.quad_xyth(nnodes_x = nx_hr,
                                                        nnodes_y = ny_hr)
            
            # Store spatial projection matrices for later reuse
            if (nx, nx_hr) in phi_projs.keys():
                phi_proj = phi_projs[(nx, nx_hr)][:]
            else:
                phi_proj = np.zeros([nx, nx_hr])
                for ii in range(0, nx):
                    for pp in range(0, nx_hr):
                        phi_proj[ii, pp] =  qd.lag_eval(xxb, ii, xxb_hr[pp])
                phi_projs[(nx, nx_hr)] = phi_proj[:]
                        
            if (ny, ny_hr) in psi_projs.keys():
                psi_proj = psi_projs[(ny, ny_hr)][:]
            else:
                psi_proj = np.zeros([ny, ny_hr])
                for jj in range(0, ny):
                    for qq in range(0, ny_hr):
                        psi_proj[jj, qq] = qd.lag_eval(yyb, jj, yyb_hr[qq])
                psi_projs[(ny, ny_hr)] = psi_proj[:]
                
            
            col_err = 0.
            
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, th1] = cell.pos[:]
                    dth = th1 - th0
                    
                    [nth]    = cell.ndofs[:]
                    [nth_hr] = mesh_hr.cols[col_key].cells[cell_key].ndofs[:]

                    [_, _, _, _, thb,    _] = qd.quad_xyth(nnodes_th = nth)
                    [_, _, _, _, thb_hr, _] = qd.quad_xyth(nnodes_th = nth_hr)
                    
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
                                xsi_proj[aa, rr] = qd.lag_eval(thb, aa, thb_hr[rr])
                        xsi_projs[(nth, nth_hr)] = xsi_proj[:]
                        
                    for pp in range(0, nx_hr):
                        for qq in range(0, ny_hr):
                            for rr in range(0, nth_hr):
                                for ii in range(0, nx):
                                    phi_ip = phi_proj[ii, pp]
                                    for jj in range(0, ny):
                                        psi_jq = psi_proj[jj, qq]
                                        for aa in range(0, nth):
                                            xsi_ar = xsi_proj[aa, rr]
                                            uh_hr_cell[pp, qq, rr] += uh_cell[ii, jj, aa] \
                                                * phi_ip * psi_jq * xsi_ar
                            
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
