import numpy as np

from .Error_Indicator import Error_Indicator
from .hp_steer import hp_steer_col, hp_steer_cell

import dg.quadrature as qd
from dg.projection import push_forward, pull_back

phi_projs = {}
psi_projs = {}
xsi_projs = {}

def anl_err(mesh, proj, anl_sol, **kwargs):
    return anl_err_hr_L2(mesh, proj, anl_sol, **kwargs)

def anl_err_hr_L2(mesh, proj, anl_sol, **kwargs):
    """
    Calculate the L2-error by cell (and column), weighted to be the relative error.
    Use a high-resolution analytioc solution and its associated quadrature.
    """
    
    default_kwargs = {'ref_col'      : True,
                      'col_ref_form' : 'hp',
                      'col_ref_kind' : 'spt',
                      'col_ref_tol'  : 0.85,
                      'ref_cell'      : True,
                      'cell_ref_form' : 'hp',
                      'cell_ref_kind' : 'ang',
                      'cell_ref_tol'  : 0.85}
    kwargs = {**default_kwargs, **kwargs}
    
    err_ind = Error_Indicator(mesh, **kwargs)
    
    col_items = sorted(mesh.cols.items())
    
    # Relative error is weighted by the integral of the analytic solution, u
    u_intg = 0.
    
    # Track maximum error(s) to calculate hp-steering only where needed
    col_max_err  = 0.
    col_ref_tol  = kwargs['col_ref_tol']
    cell_max_err = 0.
    cell_ref_tol = kwargs['cell_ref_tol']
    
    # Calculate the errors
    for col_key, col in col_items:
        if col.is_lf:
            # Column information for quadrature
            [x0, y0, xf, yf] = col.pos[:]
            [dx, dy]         = [xf - x0, yf - y0]
            [ndof_x, ndof_y] = col.ndofs[:]
            [ndof_x_hr, ndof_y_hr] = [ndof_x + 3, ndof_y + 3]
            
            # Store spatial projection matrices for later reuse
            if (ndof_x, ndof_x_hr) in phi_projs.keys():
                phi_proj = phi_projs[(ndof_x, ndof_x_hr)][:]
            else:
                [xxb,    _, _, _, _, _] = qd.quad_xyth(nnodes_x = ndof_x)
                
                [xxb_hr, _, _, _, _, _] = qd.quad_xyth(nnodes_x = ndof_x_hr)
                
                phi_proj = np.zeros([ndof_x, ndof_x_hr])
                for ii in range(0, ndof_x):
                    for pp in range(0, ndof_x_hr):
                        phi_proj[ii, pp] = qd.lag_eval(xxb, ii, xxb_hr[pp])
                phi_projs[(ndof_x, ndof_x_hr)] = phi_proj[:]
                
            if (ndof_y, ndof_y_hr) in psi_projs.keys():
                psi_proj = psi_projs[(ndof_y, ndof_y_hr)][:]
            else:
                [_, _, yyb   , _, _, _] = qd.quad_xyth(nnodes_y = ndof_y)
                
                [_, _, yyb_hr, _, _, _] = qd.quad_xyth(nnodes_y = ndof_y_hr)
                
                psi_proj = np.zeros([ndof_y, ndof_y_hr])
                for jj in range(0, ndof_y):
                    for qq in range(0, ndof_y_hr):
                        psi_proj[jj, qq] = qd.lag_eval(yyb, jj, yyb_hr[qq])
                psi_projs[(ndof_y, ndof_y_hr)] = psi_proj[:]
                
            [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = ndof_x_hr,
                                                    nnodes_y = ndof_y_hr)
            
            xxf = push_forward(x0, xf, xxb).reshape([ndof_x_hr, 1, 1])
            wx  = wx.reshape([ndof_x_hr, 1, 1])
            yyf = push_forward(y0, yf, yyb).reshape([1, ndof_y_hr, 1])
            wy  = wy.reshape([1, ndof_y_hr, 1])
                
            # Loop through cells to calculate error
            col_err    = 0.
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    # Cell information for quadrature
                    [th0, th1] = cell.pos[:]
                    dth = th1 - th0
                    [ndof_th]  = cell.ndofs[:]
                    
                    ndof_th_hr = ndof_th + 3
                    
                    # Store angular projection matrices for later reuse
                    if (ndof_th, ndof_th_hr) in xsi_projs.keys():
                        xsi_proj = xsi_projs[(ndof_th, ndof_th_hr)][:]
                    else:
                        [_, _, _, _, thb,    _] = qd.quad_xyth(nnodes_th = ndof_th)
                        
                        [_, _, _, _, thb_hr, _] = qd.quad_xyth(nnodes_th = ndof_th_hr)
                        
                        xsi_proj = np.zeros([ndof_th, ndof_th_hr])
                        for aa in range(0, ndof_th):
                            for rr in range(0, ndof_th_hr):
                                xsi_proj[aa, rr] = qd.lag_eval(thb, aa, thb_hr[rr])
                        xsi_projs[(ndof_th, ndof_th_hr)] = xsi_proj[:]
                        
                    [_, _, _, _, thb, wth] = qd.quad_xyth(nnodes_th = ndof_th_hr)
                    thf = push_forward(th0, th1, thb).reshape([1, 1, ndof_th_hr])
                    wth = wth.reshape([1, 1, ndof_th_hr])
                    
                    # Calculate error
                    u_cell  = anl_sol(xxf, yyf, thf)
                    uh_cell = proj.cols[col_key].cells[cell_key].vals
                    uh_cell_hr = np.zeros([ndof_x_hr, ndof_y_hr, ndof_th_hr])
                    for pp in range(0, ndof_x_hr):
                        for qq in range(0, ndof_y_hr):
                            for rr in range(0, ndof_th_hr):
                                for ii in range(0, ndof_x):
                                    phi_ip = phi_proj[ii, pp]
                                    for jj in range(0, ndof_y):
                                        psi_jq = psi_proj[jj, qq]
                                        for aa in range(0, ndof_th):
                                            xsi_ar = xsi_proj[aa, rr]
                                            
                                            uh_ija = uh_cell[ii, jj, aa]
                                            
                                            uh_cell_hr[pp, qq, rr] += uh_ija * phi_ip * psi_jq * xsi_ar
                    
                    cell_err = (dx * dy * dth / 8.) * np.sum(wx * wy * wth * (u_cell - uh_cell_hr)**2)
                    col_err += cell_err
                    
                    u_intg  += (dx * dy * dth / 8.) * np.sum(wx * wy * wth * (u_cell)**2)
                    
                    if kwargs['ref_cell']:
                        err_ind.cols[col_key].cells[cell_key].err = cell_err
                        cell_max_err = max(cell_max_err, cell_err)
            
            if kwargs['ref_col']:
                err_ind.cols[col_key].err = col_err
                col_max_err = max(col_max_err, col_err)

    # Weight errors to be relative, and calculate hp-steering criteria
    if kwargs['ref_col']:
        col_max_err  /= u_intg
        col_ref_thrsh = col_ref_tol * col_max_err
        err_ind.col_max_err = col_max_err

    if kwargs['ref_cell']:
        cell_max_err  /= u_intg
        cell_ref_thrsh = cell_ref_tol * cell_max_err
        err_ind.cell_max_err = cell_max_err
            
    # Weight to be relative error
    for col_key, col in col_items:
        if col.is_lf:
            if kwargs['ref_col']: # If we're refining columns
                err_ind.cols[col_key].err /= u_intg
                if err_ind.cols[col_key].err >= col_ref_thrsh: # Does this one need to be refined?
                    if err_ind.cols[col_key].ref_form == 'hp': # Does the form of refinement need to be chosen?
                        err_ind.cols[col_key].ref_form = hp_steer_col(mesh, proj, col_key)
                else: # Needn't be refined
                    err_ind.cols[col_key].ref_form = None
                        
                
            if kwargs['ref_cell']: # If we're refining cells
                cell_items = sorted(col.cells.items())
                for cell_key, cell in cell_items:
                    if cell.is_lf:
                        err_ind.cols[col_key].cells[cell_key].err /= u_intg
                        
                        if err_ind.cols[col_key].cells[cell_key].err >= cell_ref_thrsh: # Does this one need to be refined?
                            if err_ind.cols[col_key].cells[cell_key].ref_form == 'hp': # Does the form of refinement need to be chosen?
                                err_ind.cols[col_key].cells[cell_key].ref_form = \
                                    hp_steer_cell(mesh, proj, col_key, cell_key)
                        else: # Needn't be refined
                            err_ind.cols[col_key].cells[cell_key].ref_form = None
                            
    return err_ind

def anl_err_L2(mesh, proj, anl_sol, **kwargs):
    """
    Calculate the L2-error by cell (and column), weighted to be the relative error.
    """
    
    default_kwargs = {'ref_col'      : True,
                      'col_ref_form' : 'hp',
                      'col_ref_kind' : 'spt',
                      'col_ref_tol'  : 0.85,
                      'ref_cell'      : True,
                      'cell_ref_form' : 'hp',
                      'cell_ref_kind' : 'ang',
                      'cell_ref_tol'  : 0.85}
    kwargs = {**default_kwargs, **kwargs}
    
    err_ind = Error_Indicator(mesh, **kwargs)
    
    col_items = sorted(mesh.cols.items())
    
    # Relative error is weighted by the integral of the analytic solution, u
    u_intg = 0.
    
    # Track maximum error(s) to calculate hp-steering only where needed
    col_max_err  = 0.
    col_ref_tol  = kwargs['col_ref_tol']
    cell_max_err = 0.
    cell_ref_tol = kwargs['cell_ref_tol']
    
    # Calculate the errors
    for col_key, col in col_items:
        if col.is_lf:
            # Column information for quadrature
            [x0, y0, xf, yf] = col.pos[:]
            [dx, dy]         = [xf - x0, yf - y0]
            [ndof_x, ndof_y] = col.ndofs[:]
            
            [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                    nnodes_y = ndof_y)
            
            xxf = push_forward(x0, xf, xxb).reshape(ndof_x, 1, 1)
            wx  = wx.reshape(ndof_x, 1, 1)
            yyf = push_forward(y0, yf, yyb).reshape(1, ndof_y, 1)
            wy  = wy.reshape(1, ndof_y, 1)
            
            # Loop through cells to calculate error
            col_err    = 0.
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    # Cell information for quadrature
                    [th0, thf] = cell.pos[:]
                    dth = thf - th0
                    [ndof_th]  = cell.ndofs[:]
                    
                    [_, _, _, _, thb, wth] = qd.quad_xyth(nnodes_th = ndof_th)
                    
                    thf = push_forward(th0, thf, thb).reshape(1, 1, ndof_th)
                    wth = wth.reshape(1, 1, ndof_th)
                    
                    # Calculate error
                    uh_cell = proj.cols[col_key].cells[cell_key].vals
                    u_cell  = anl_sol(xxf, yyf, thf)
                    
                    cell_err = np.sum((dx * dy * dth / 8.) * wx * wy * wth * (u_cell - uh_cell)**2)
                    col_err += cell_err
                    
                    u_intg  += np.sum((dx * dy * dth / 8.) * wx * wy * wth * (u_cell)**2)
                    
                    if kwargs['ref_cell']:
                        err_ind.cols[col_key].cells[cell_key].err = cell_err
                        cell_max_err = max(cell_max_err, cell_err)
            
            if kwargs['ref_col']:
                err_ind.cols[col_key].err = col_err
                col_max_err = max(col_max_err, col_err)

    # Weight errors to be relative, and calculate hp-steering criteria
    if kwargs['ref_col']:
        col_max_err  /= u_intg
        col_ref_thrsh = col_ref_tol * col_max_err
        err_ind.col_max_err = col_max_err

    if kwargs['ref_cell']:
        cell_max_err  /= u_intg
        cell_ref_thrsh = cell_ref_tol * cell_max_err
        err_ind.cell_max_err = cell_max_err
            
    # Weight to be relative error
    for col_key, col in col_items:
        if col.is_lf:
            if kwargs['ref_col']: # If we're refining columns
                err_ind.cols[col_key].err /= u_intg
                if err_ind.cols[col_key].err >= col_ref_thrsh: # Does this one need to be refined?
                    if err_ind.cols[col_key].ref_form == 'hp': # Does the form of refinement need to be chosen?
                        err_ind.cols[col_key].ref_form = hp_steer_col(mesh, proj, col_key)
                else: # Needn't be refined
                    err_ind.cols[col_key].ref_form = None
                        
                
            if kwargs['ref_cell']: # If we're refining cells
                cell_items = sorted(col.cells.items())
                for cell_key, cell in cell_items:
                    if cell.is_lf:
                        err_ind.cols[col_key].cells[cell_key].err /= u_intg
                        
                        if err_ind.cols[col_key].cells[cell_key].err >= cell_ref_thrsh: # Does this one need to be refined?
                            if err_ind.cols[col_key].cells[cell_key].ref_form == 'hp': # Does the form of refinement need to be chosen?
                                err_ind.cols[col_key].cells[cell_key].ref_form = \
                                    hp_steer_cell(mesh, proj, col_key, cell_key)
                        else: # Needn't be refined
                            err_ind.cols[col_key].cells[cell_key].ref_form = None
                            
    return err_ind

def anl_err_max(mesh, proj, anl_sol, **kwargs):
    """
    Here we calculate the max-error by cell (and column), weighted to be the
    relative error.
    """
    
    default_kwargs = {'ref_col'      : True,
                      'col_ref_form' : 'hp',
                      'col_ref_kind' : 'spt',
                      'col_ref_tol'  : 0.85,
                      'ref_cell'      : True,
                      'cell_ref_form' : 'hp',
                      'cell_ref_kind' : 'ang',
                      'cell_ref_tol'  : 0.85}
    kwargs = {**default_kwargs, **kwargs}
    
    err_ind = Error_Indicator(mesh, **kwargs)
    
    col_items = sorted(mesh.cols.items())
    
    # Relative error is weighted by the maximum value of the anlytic solution, u
    u_max = 0.
    
    # Track maximum error(s) to calculate hp-steering only where needed
    col_max_err  = 0.
    col_ref_tol  = kwargs['col_ref_tol']
    cell_max_err = 0.
    cell_ref_tol = kwargs['cell_ref_tol']
    
    # Calculate the errors
    for col_key, col in col_items:
        if col.is_lf:
            # Loop through cells to calculate error
            col_err    = 0.
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:                    
                    # Calculate error
                    uh_cell = proj.cols[col_key].cells[cell_key].vals
                    u_cell  = anl_sol(xxf, yyf, thf)
                    
                    cell_err = np.amax(np.abs(u_cell - uh_cell))
                    col_err  = max(col_err, cell_err)
                    
                    u_max    = max(u_max, np.amax(np.abs(u_cell)))
                    
                    if kwargs['ref_cell']:
                        err_ind.cols[col_key].cells[cell_key].err = cell_err
                        cell_max_err = max(cell_max_err, cell_err)
            
            if kwargs['ref_col']:
                err_ind.cols[col_key].err = col_err
                col_max_err = max(col_max_err, col_err)

    # Weight errors to be relative, and calculate hp-steering criteria
    if kwargs['ref_col']:
        col_max_err  /= u_max
        col_ref_thrsh = col_ref_tol * col_max_err
        err_ind.col_max_err = col_max_err

    if kwargs['ref_cell']:
        cell_max_err  /= u_max
        cell_ref_thrsh = cell_ref_tol * cell_max_err
        err_ind.cell_max_err = cell_max_err
            
    # Weight to be relative error
    for col_key, col in col_items:
        if col.is_lf:
            err_ind.cols[col_key].err /= u_max

            if kwargs['ref_col']: # If we're refining columns
                if err_ind.cols[col_key].err >= col_ref_thrsh: # Does this one need to be refined?
                    if err_ind.cols[col_key].ref_form == 'hp': # Does the form of refinement need to be chosen?
                        err_ind.cols[col_key].ref_form = hp_steer_col(mesh, proj, col_key)
                else: # Needn't be refined
                    err_ind.cols[col_key].ref_form = None
                        
                
            if kwargs['ref_cell']: # If we're refining cells
                cell_items = sorted(col.cells.items())
                for cell_key, cell in cell_items:
                    if cell.is_lf:
                        err_ind.cols[col_key].cells[cell_key].err /= u_max
                        
                        if err_ind.cols[col_key].cells[cell_key].err >= cell_ref_thrsh: # Does this one need to be refined?
                            if err_ind.cols[col_key].cells[cell_key].ref_form == 'hp': # Does the form of refinement need to be chosen?
                                err_ind.cols[col_key].cells[cell_key].ref_form = \
                                    hp_steer_cell(mesh, proj, col_key, cell_key)
                        else: # Needn't be refined
                            err_ind.cols[col_key].cells[cell_key].ref_form = None
                            
    return err_ind
