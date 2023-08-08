import numpy as np

from .Error_Indicator import Error_Indicator
from .hp_steer        import hp_steer_cell, hp_steer_col

import dg.quadrature as qd

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

def cell_jump_err_old(mesh, proj, **kwargs):

    default_kwargs = {'ref_col'      : False,
                      'col_ref_form' : None,
                      'col_ref_kind' : None,
                      'col_ref_tol'  : None,
                      'ref_cell'      : True,
                      'cell_ref_form' : 'hp',
                      'cell_ref_kind' : 'ang',
                      'cell_ref_tol'  : 0.85}
    kwargs = {**default_kwargs, **kwargs}
    
    err_ind = Error_Indicator(mesh, **kwargs)
    col_nref = 0
    cell_nref = 0
    
    col_items = sorted(mesh.cols.items())
    
    # Track maximum error(s) to calculate hp-steering only where needed
    col_max_err  = 0.
    col_ref_tol  = kwargs['col_ref_tol']
    cell_max_err  = 0.
    cell_ref_tol  = kwargs['cell_ref_tol']
    
    # Begin by integrating each column with respect to theta
    cell_intg_xys = {}
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    cell_intg_xys[(col_key, cell_key)] = \
                        intg_cell_bdry_xy(mesh, proj, col_key, cell_key)
            
    # Once we have spatially-integrated along angular faces of each cell, we
    # calculate the jumps
    col_errs = []
    cell_errs = []
    for col_key, col in col_items:
        if col.is_lf:
            # Column information for weighting
            [x0, y0, xf, yf] = col.pos[:]
            [dx, dy]         = [xf - x0, yf - y0]
            [ndof_x, ndof_y] = col.ndofs[:]
            
            dA = dx * dy

            if kwargs['ref_col']:
                col_err = 0.
            
            #Loop through cells to calculate error
            cell_items = sorted(col.cells.items())
            for cell_key_0, cell_0 in cell_items:
                if cell_0.is_lf:
                    cell_err = 0.
                    
                    for F in range(0, 2):
                        cell_key_1 = cell_0.nhbr_keys[F]
                        
                        cell_intg_xy_0 = cell_intg_xys[(col_key, cell_key_0)][F]
                        cell_intg_xy_1 = cell_intg_xys[(col_key, cell_key_1)][(F+1)%2]
                        
                        cell_err += ((1. / dA) * (cell_intg_xy_0 - cell_intg_xy_1))**2
                        
                    cell_err = np.sqrt(cell_err)
                    cell_max_err = max(cell_max_err, cell_err)
                    if kwargs['ref_cell']:
                        err_ind.cols[col_key].cells[cell_key_0].err = cell_err
                        cell_errs += [[col_key, cell_key_0, cell_err]]
                        
                    if kwargs['ref_col']:
                        col_err += cell_err
                        
            if kwargs['ref_col']:
                err_ind.cols[col_key].err = col_err
                col_errs += [[col_key, col_err]]
                col_max_err = max(col_max_err, col_err)

    """
    # Sort errors, then pick top (1-ref_tol) percent of them to refine
    if kwargs['ref_col']:
        err_ind.col_max_err = col_max_err
        col_errs    = np.array(col_errs)
        mask        = np.argsort(col_errs[:, 1])
        col_ref_ord = col_errs[mask, 0]
        ncol        = np.size(col_errs)
        ncol_to_ref = int(np.ceil((1. - col_ref_tol) * ncol))

    if kwargs['ref_cell']:
        err_ind.cell_max_err = cell_max_err
        cell_errs    = np.array(cell_errs)
        mask         = np.argsort(-cell_errs[:, 2])
        cell_ref_ord = cell_errs[mask, 0:2].astype('int32')
        ncell        = np.size(cell_errs[:, 0])
        ncell_to_ref = int(np.ceil((1. - cell_ref_tol) * ncell))
        
        
    if kwargs['ref_col']:
        for cc in range(0, ncol_to_ref):
            col_key = col_ref_ord[cc]
            col = mesh.cols[col_key]
            if col.is_lf:
                if err_ind.cols[col_key].ref_form == 'hp':
                        err_ind.cols[col_key].ref_form = hp_steer_col(mesh, proj, col_key)
                        
        for cc in range(ncol_to_ref, ncol):
            col_key = col_errs[cc]
            col = mesh.cols[col_key]
            if col.is_lf:
                err_ind.cols[col_key].ref_form = None

    if kwargs['ref_cell']:
        for kk in range(0, ncell_to_ref):
            col_key = cell_ref_ord[kk, 0]
            cell_key = cell_ref_ord[kk, 1]
            col = mesh.cols[col_key]
            if col.is_lf:
                cell = col.cells[cell_key]
                if cell.is_lf:
                    if err_ind.cols[col_key].cells[cell_key].ref_form == 'hp':
                            err_ind.cols[col_key].cells[cell_key].ref_form = hp_steer_cell(mesh, proj, col_key, cell_key)
                            
        for kk in range(ncell_to_ref, ncell):
            col_key = cell_ref_ord[kk, 0]
            cell_key = cell_ref_ord[kk, 1]
            col = mesh.cols[col_key]
            if col.is_lf:
                cell = col.cells[cell_key]
                if cell.is_lf:
                    err_ind.cols[col_key].cells[cell_key].ref_form = None
    """
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
                if  col_err >= col_ref_thrsh:
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
                    if cell_err >= cell_ref_thrsh:
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

def intg_cell_bdry_xy(mesh, proj, col_key, cell_key):
    '''
    Integrate the angular-top and -bot in space.
    '''
    col = mesh.cols[col_key]
    if col.is_lf:
        [x0, y0, xf, yf] = col.pos[:]
        [dx, dy]         = [xf - x0, yf - y0]
        [ndof_x, ndof_y] = col.ndofs[:]
        
        [_, w_x, _, w_y, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                              nnodes_y = ndof_y)

        w_x = w_x.reshape(ndof_x, 1, 1)
        w_y = w_y.reshape(1, ndof_y, 1)
        
        dcoeff = (dx * dy / 4.)
        
        proj_col  = proj.cols[col_key]
        
        cell = col.cells[cell_key]
        if cell.is_lf:
            [ndof_th]  = cell.ndofs
            
            proj_cell  = proj_col.cells[cell_key]
            uh_cell    = proj_cell.vals

            [_, _, _, _, thb, _] = qd.quad_xyth(nnodes_th = ndof_th)
                    
            uh_th = dcoeff * np.sum(w_x * w_y * uh_cell, axis = (0, 1))
        
            # Store spatial integral along each angular face
            # F = 0 => Bottom
            cell_intg_xy = [0, 0]
            
            for F in range(0, 2):
                if F == 0:
                    thb_rr = -1.
                else:
                    thb_rr = 1.
                    
                uh_end = 0.
                for aa in range(0, ndof_th):
                    uh_end += uh_th[aa] * qd.lag_eval(thb, aa, thb_rr)
                cell_intg_xy[F] = uh_end
                                
        return cell_intg_xy
