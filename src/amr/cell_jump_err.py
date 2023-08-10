import numpy as np

from .Error_Indicator import Error_Indicator
from .hp_steer        import hp_steer_cell, hp_steer_col

import dg.quadrature as qd

def cell_jump_err_new(mesh, proj, **kwargs):
    # Skip calculating the jumps between the inflow and outflow boundary
    default_kwargs = {'ref_col'      : False,
                      'col_ref_form' : None,
                      'col_ref_kind' : None,
                      'col_ref_tol'  : None,
                      'ref_cell'      : True,
                      'cell_ref_form' : 'hp',
                      'cell_ref_kind' : 'ang',
                      'cell_ref_tol'  : 0.85}
    kwargs = {**default_kwargs, **kwargs}
    
    err_ind   = Error_Indicator(mesh, **kwargs)
    col_nref  = 0
    cell_nref = 0
    
    col_items = sorted(mesh.cols.items())

    rng = np.random.default_rng()
    
    # Track maximum error(s) to calculate hp-steering only where needed
    col_max_err   = 0.
    col_ref_tol   = kwargs['col_ref_tol']
    cell_max_err  = 0.
    cell_ref_tol  = kwargs['cell_ref_tol']
    
    # Begin by getting the radiation field at the top and bottom of each cell
    cell_tops = {}
    cell_bots = {}
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        [nx, ny] = col.ndofs[:]
        if col.is_lf:
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [nth]      = cell.ndofs[:]
                    
                    [_, _, _, _, thb, _] = qd.quad_xyth(nnodes_th = nth)
                    
                    cell_vals = proj.cols[col_key].cells[cell_key].vals[:,:,:]
                    cell_top = np.zeros([nx, ny])
                    cell_bot = np.zeros([nx, ny])
                    
                    # Pull-back top is 1, pull-back bot is -1
                    for aa in range(0, nth):
                        cell_top += cell_vals[:,:,aa] * qd.lag_eval(thb, aa, 1)
                        cell_bot += cell_vals[:,:,aa] * qd.lag_eval(thb, aa, -1)
                        
                    cell_tops[(col_key, cell_key)] = cell_top[:,:]
                    cell_bots[(col_key, cell_key)] = cell_bot[:,:]
                    
    # Once we have the value of the radiation field at the top and bottom of
    # each cell, we can calculate the jumps and integrate those spatially
    for col_key, col in col_items:
        if col.is_lf:
            # Column information for weighting
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy]         = [x1 - x0, y1 - y0]
            dA               = dx * dy
            [nx, ny]         = col.ndofs[:]
            
            [_, wx, _, wy, _, _] = qd.quad_xyth(nnodes_x = nx, nnodes_y = ny)
            wx = wx.reshape([nx, 1])
            wy = wy.reshape([1, ny])
            
            col_nhbr_keys = col.nhbr_keys[:]
            
            # Which spatial bounday(ies) the element is on
            rs_bdry = ((col_nhbr_keys[0][0] is None) and (col_nhbr_keys[0][1] is None))
            ts_bdry = ((col_nhbr_keys[1][0] is None) and (col_nhbr_keys[1][1] is None))
            ls_bdry = ((col_nhbr_keys[2][0] is None) and (col_nhbr_keys[2][1] is None))
            bs_bdry = ((col_nhbr_keys[3][0] is None) and (col_nhbr_keys[3][1] is None))
            
            # Get the interfaces between inflow and outflow
            if rs_bdry and ts_bdry:
                io_itfs = [0., np.pi / 2.]
            elif rs_bdry and bs_bdry:
                io_itfs = [0., 3. * np.pi / 2.]
            elif ls_bdry and ts_bdry:
                io_itfs = [np.pi / 2., np.pi]
            elif ls_bdry and bs_bdry:
                io_itfs = [np.pi, 3. * np.pi / 2.]
            elif rs_bdry or ls_bdry:
                io_itfs = [np.pi / 2., 3. * np.pi / 2.]
            elif ts_bdry or bs_bdry:
                io_itfs = [0., np.pi]
            
            if kwargs['ref_col']:
                col_err = 0.
            
            # Cell_0 is self
            cell_items = sorted(col.cells.items())
            for cell_key_0, cell_0 in cell_items:
                if cell_0.is_lf:
                    [th0, th1] = cell.pos[:]
                    # If on edge between inflow and outflow boundary,
                    # set values at that boundary to zero
                    
                    # Jump with lower-neighbor information
                    cell_bot_0   = cell_bots[(col_key, cell_key_0)]
                    nhbr_low_key = cell_0.nhbr_keys[0]
                    nhbr_low_top = cell_tops[(col_key, nhbr_low_key)]

                    # Jump with upper-neighbor information
                    cell_top_0   = cell_tops[(col_key, cell_key_0)]
                    nhbr_up_key  = cell_0.nhbr_keys[1]
                    nhbr_up_bot  = cell_bots[(col_key, nhbr_up_key)]
                    
                    # If on inflow-outflow interface, ignore that jump
                    if th0 in io_itfs:
                        if rs_bdry:
                            cell_bot_0[nx-1, :]   = 0.
                            nhbr_low_top[nx-1, :] = 0.
                        if ls_bdry:
                            cell_bot_0[0, :]   = 0.
                            nhbr_low_top[0, :] = 0.
                        if ts_bdry:
                            cell_bot_0[:, ny-1]   = 0.
                            nhbr_low_top[:, ny-1] = 0.
                        if bs_bdry:
                            cell_bot_0[:, 0]   = 0.
                            nhbr_low_top[:, 0] = 0.
                    if th1 in io_itfs:
                        if rs_bdry:
                            cell_top_0[nx-1, :]  = 0.
                            nhbr_up_bot[nx-1, :] = 0.
                        if ls_bdry:
                            cell_top_0[0, :]  = 0.
                            nhbr_up_bot[0, :] = 0.
                        if ts_bdry:
                            cell_top_0[:, ny-1]  = 0.
                            nhbr_up_bot[:, ny-1] = 0.
                        if bs_bdry:
                            cell_top_0[:, 0]  = 0.
                            nhbr_up_bot[:, 0] = 0.
                            
                    cell_jump_low = (cell_bot_0 - nhbr_low_top)**2
                    cell_jump_up  = (cell_top_0 - nhbr_up_bot)**2

                    # For integral, we multiply by dA / 4.
                    # For mean, we divide integral by 1 / dA.
                    # Hence, just divise by 4.
                    # The 0.5 arise from the fact that we take the mean of all
                    # the jumps, and there are two jumps.
                    cell_jump = (1. / 4.) * np.sum(wx * wy * 0.5 * (cell_jump_low + cell_jump_up))
                    
                    cell_err = np.sqrt(cell_jump)
                    cell_max_err = max(cell_max_err, cell_err)
                    if kwargs['ref_cell']:
                        err_ind.cols[col_key].cells[cell_key_0].err = cell_err
                        
                    if kwargs['ref_col']:
                        col_err += cell_err
                        
            if kwargs['ref_col']:
                err_ind.cols[col_key].err = col_err
                col_errs += [[col_key, col_err]]
                col_max_err = max(col_max_err, col_err)
                
    # Refine if error is at least tol*max_err
    if kwargs['ref_col']:
        err_ind.col_max_err = col_max_err
        col_ref_thrsh = col_ref_tol * col_max_err
        
    if kwargs['ref_cell']:
        err_ind.cell_max_err = cell_max_err
        cell_ref_thrsh = cell_ref_tol * cell_max_err
        
    for col_key, col in col_items:
        if col.is_lf:
            if kwargs['ref_col']:
                col_err = err_ind.cols[col_key].err
                #p = col_err / col_max_err
                #to_ref = rng.choice([True, False], size = 1, p = (p, 1 - p))[0]
                if col_err >= col_ref_thrsh:
                #if to_ref:
                    err_ind.avg_col_ref_err += col_err
                    col_nref += 1
                    if err_ind.cols[col_key].ref_form == 'hp':
                        err_ind.cols[col_key].ref_form = hp_steer_col(mesh, proj, col_key)
                else:
                    err_ind.cols[col_key].ref_form = None
                
            if kwargs['ref_cell']:
                cell_items = sorted(col.cells.items())
                for cell_key, cell in cell_items:
                    cell_err = err_ind.cols[col_key].cells[cell_key].err
                    #p = cell_err / cell_max_err
                    #to_ref = rng.choice([True, False], size = 1, p = (p, 1 - p))[0]
                    if cell_err >= cell_ref_thrsh:
                    #if to_ref:
                        err_ind.avg_cell_ref_err += cell_err
                        cell_nref += 1
                        if err_ind.cols[col_key].cells[cell_key].ref_form == 'hp':
                            err_ind.cols[col_key].cells[cell_key].ref_form = hp_steer_cell(mesh, proj, col_key, cell_key)
                    else:
                        err_ind.cols[col_key].cells[cell_key].ref_form = None
    if kwargs['ref_col']:
        err_ind.avg_col_ref_err /= col_nref
        
    if kwargs['ref_cell']:
        err_ind.avg_cell_ref_err /= cell_nref
        
    return err_ind

def cell_jump_err(mesh, proj, **kwargs):

    default_kwargs = {'ref_col'      : False,
                      'col_ref_form' : None,
                      'col_ref_kind' : None,
                      'col_ref_tol'  : None,
                      'ref_cell'      : True,
                      'cell_ref_form' : 'hp',
                      'cell_ref_kind' : 'ang',
                      'cell_ref_tol'  : 0.85}
    kwargs = {**default_kwargs, **kwargs}
    
    err_ind   = Error_Indicator(mesh, **kwargs)
    col_nref  = 0
    cell_nref = 0
    
    col_items = sorted(mesh.cols.items())

    rng = np.random.default_rng()
    
    # Track maximum error(s) to calculate hp-steering only where needed
    col_max_err   = 0.
    col_ref_tol   = kwargs['col_ref_tol']
    cell_max_err  = 0.
    cell_ref_tol  = kwargs['cell_ref_tol']
    
    # Begin by getting the radiation field at the top and bottom of each cell
    cell_tops = {}
    cell_bots = {}
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        [nx, ny] = col.ndofs[:]
        if col.is_lf:
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [nth]      = cell.ndofs[:]
                    
                    [_, _, _, _, thb, _] = qd.quad_xyth(nnodes_th = nth)
                    
                    cell_vals = proj.cols[col_key].cells[cell_key].vals[:,:,:]
                    cell_top = np.zeros([nx, ny])
                    cell_bot = np.zeros([nx, ny])
                    
                    # Pull-back top is 1, pull-back bot is -1
                    for aa in range(0, nth):
                        cell_top += cell_vals[:,:,aa] * qd.lag_eval(thb, aa, 1)
                        cell_bot += cell_vals[:,:,aa] * qd.lag_eval(thb, aa, -1)
                        
                    cell_tops[(col_key, cell_key)] = cell_top[:,:]
                    cell_bots[(col_key, cell_key)] = cell_bot[:,:]
                    
    # Once we have the value of the radiation field at the top and bottom of
    # each cell, we can calculate the jumps and integrate those spatially
    for col_key, col in col_items:
        if col.is_lf:
            # Column information for weighting
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy]         = [x1 - x0, y1 - y0]
            dA               = dx * dy
            [nx, ny]         = col.ndofs[:]
            
            [_, wx, _, wy, _, _] = qd.quad_xyth(nnodes_x = nx, nnodes_y = ny)
            wx = wx.reshape([nx, 1])
            wy = wy.reshape([1, ny])
            
            if kwargs['ref_col']:
                col_err = 0.
            
            # Cell_0 is self
            cell_items = sorted(col.cells.items())
            for cell_key_0, cell_0 in cell_items:
                if cell_0.is_lf:
                    # Jump with lower-neighbor
                    cell_bot_0   = cell_bots[(col_key, cell_key_0)]
                    nhbr_low_key = cell_0.nhbr_keys[0]
                    nhbr_low_top = cell_tops[(col_key, nhbr_low_key)]
                    cell_jump_low = (cell_bot_0 - nhbr_low_top)**2
                    
                    # Jump with upper-neighbor
                    cell_top_0   = cell_tops[(col_key, cell_key_0)]
                    nhbr_up_key  = cell_0.nhbr_keys[1]
                    nhbr_up_bot  = cell_bots[(col_key, nhbr_up_key)]
                    cell_jump_up = (cell_top_0 - nhbr_up_bot)**2

                    # For integral, we multiply by dA / 4.
                    # For mean, we divide integral by 1 / dA.
                    # Hence, just divise by 4.
                    # The 0.5 arise from the fact that we take the mean of all
                    # the jumps, and there are two jumps.
                    cell_jump = (1. / 4.) * np.sum(wx * wy * 0.5 * (cell_jump_low + cell_jump_up))
                    
                    cell_err = np.sqrt(cell_jump)
                    cell_max_err = max(cell_max_err, cell_err)
                    if kwargs['ref_cell']:
                        err_ind.cols[col_key].cells[cell_key_0].err = cell_err
                        
                    if kwargs['ref_col']:
                        col_err += cell_err
                        
            if kwargs['ref_col']:
                err_ind.cols[col_key].err = col_err
                col_errs += [[col_key, col_err]]
                col_max_err = max(col_max_err, col_err)
                
    # Refine if error is at least tol*max_err
    if kwargs['ref_col']:
        err_ind.col_max_err = col_max_err
        col_ref_thrsh = col_ref_tol * col_max_err
        
    if kwargs['ref_cell']:
        err_ind.cell_max_err = cell_max_err
        cell_ref_thrsh = cell_ref_tol * cell_max_err
        
    for col_key, col in col_items:
        if col.is_lf:
            if kwargs['ref_col']:
                col_err = err_ind.cols[col_key].err
                #p = col_err / col_max_err
                #to_ref = rng.choice([True, False], size = 1, p = (p, 1 - p))[0]
                if col_err >= col_ref_thrsh:
                #if to_ref:
                    err_ind.avg_col_ref_err += col_err
                    col_nref += 1
                    if err_ind.cols[col_key].ref_form == 'hp':
                        err_ind.cols[col_key].ref_form = hp_steer_col(mesh, proj, col_key)
                else:
                    err_ind.cols[col_key].ref_form = None
                
            if kwargs['ref_cell']:
                cell_items = sorted(col.cells.items())
                for cell_key, cell in cell_items:
                    cell_err = err_ind.cols[col_key].cells[cell_key].err
                    #p = cell_err / cell_max_err
                    #to_ref = rng.choice([True, False], size = 1, p = (p, 1 - p))[0]
                    if cell_err >= cell_ref_thrsh:
                    #if to_ref:
                        err_ind.avg_cell_ref_err += cell_err
                        cell_nref += 1
                        if err_ind.cols[col_key].cells[cell_key].ref_form == 'hp':
                            err_ind.cols[col_key].cells[cell_key].ref_form = hp_steer_cell(mesh, proj, col_key, cell_key)
                    else:
                        err_ind.cols[col_key].cells[cell_key].ref_form = None
    if kwargs['ref_col']:
        err_ind.avg_col_ref_err /= col_nref
        
    if kwargs['ref_cell']:
        err_ind.avg_cell_ref_err /= cell_nref
        
    return err_ind
