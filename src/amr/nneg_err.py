import numpy as np

import dg.projection as proj
import dg.quadrature as qd

from .Error_Indicator import Error_Indicator
from .hp_steer        import hp_steer_col, hp_steer_cell

def nneg_err(mesh, proj, **kwargs):
    return nneg_err_abs(mesh, proj, **kwargs)

def nneg_err_abs(mesh, proj, **kwargs):
    """
    Refine all with a negative value
    """
    default_kwargs = {"ref_col"      : True,
                      "col_ref_form" : "hp",
                      "col_ref_kind" : "spt",
                      "col_ref_tol"  : 0.0,
                      "ref_cell"      : True,
                      "cell_ref_form" : "hp",
                      "cell_ref_kind" : "ang",
                      "cell_ref_tol"  : 0.0}
    kwargs = {**default_kwargs, **kwargs}
    
    err_ind = Error_Indicator(mesh, **kwargs)
    
    col_items = sorted(mesh.cols.items())

    # Assign error of 1. to cols, cells with a negative value
    for col_key, col in col_items:
        if col.is_lf:
            if kwargs["ref_col"]:
                col_has_neg = False
                
            # Loop through cells
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    uh_cell = proj.cols[col_key].cells[cell_key].vals
                    if kwargs["ref_cell"]:
                        cell_has_neg = np.any(uh_cell < kwargs["cell_ref_tol"])
                        cell_err = int(cell_has_neg)
                        
                    if kwargs["ref_col"]:
                        col_has_neg = col_has_neg or np.any(uh_cell < kwargs["col_ref_tol"])
                    
                    if kwargs["ref_cell"]:
                        err_ind.cols[col_key].cells[cell_key].err = cell_err
                        if cell_has_neg:
                            if kwargs["cell_ref_form"] == "hp":
                                err_ind.cols[col_key].cells[cell_key].ref_form = hp_steer_cell(mesh, proj, col_key, cell_key)
                        else:
                            err_ind.cols[col_key].cells[cell_key].ref_form = None
                            
            if kwargs["ref_col"]:
                err_ind.cols[col_key].err = int(col_has_neg)
                if col_has_neg:
                    if kwargs["col_ref_form"] == "hp":
                        err_ind.cols[col_key].ref_form = hp_steer_col(mesh, proj, col_key)
                else:
                    err_ind.cols[col_key].ref_form = None
                        
    if kwargs["ref_col"]:
        err_ind.col_max_err = 1.
        
    if kwargs["ref_cell"]:
        err_ind.cell_max_err = 1.
            
    return err_ind

def nneg_err_ang(mesh, proj, **kwargs):
    """
    Calculate spatial integral in each spatio-angular element. If beyond negative
    tolerance, mark to refine.
    """
    
    # If integral of proejction has values less than the (negative) tolerance,
    # refine
    default_kwargs = {"ref_col"      : False,
                      "col_ref_form" : None,
                      "col_ref_kind" : None,
                      "col_ref_tol"  : None,
                      "ref_cell"      : True,
                      "cell_ref_form" : "hp",
                      "cell_ref_kind" : "ang",
                      "cell_ref_tol"  : 0.0}
    kwargs = {**default_kwargs, **kwargs}
    
    err_ind = Error_Indicator(mesh, **kwargs)
    col_nref = 0
    cell_nref = 0
    
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy]         = [x1 - x0, y1 - y0]
            [nx, ny]         = col.ndofs[:]
            
            [_, wx, _, wy, _, _] = qd.quad_xyth(nnodes_x = nx, nnodes_y = ny)
            wx  = wx.reshape([nx, 1, 1])
            wy  = wy.reshape([1, ny, 1])
            
            if kwargs["ref_col"]:
                col_err = 0
            
            # Loop through cells
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, th1] = cell.pos[:]
                    dth        = th1 - th0
                    [nth]      = cell.ndofs[:]
                    
                    [_, _, _, _, _, wth] = qd.quad_xyth(nnodes_th = nth)
                    wth  = wth.reshape([1, 1, nth])
                    
                    uh_cell      = proj.cols[col_key].cells[cell_key].vals[:,:,:]
                    uh_cell_intg = (1 / dth ) * (dx * dy * dth / 8.) * np.sum(wx * wy * uh_cell)
                    
                    if kwargs["ref_cell"]:
                        cell_err = min(np.amin(uh_cell_intg), 0.)
                        
                    if kwargs["ref_col"]:
                        col_err = min(np.amin(uh_cell_intg), col_err)
                    
                    if kwargs["ref_cell"]:
                        err_ind.cell_max_err = min(cell_err, err_ind.cell_max_err)
                        err_ind.cols[col_key].cells[cell_key].err = cell_err
                        if cell_err < kwargs["cell_ref_tol"]:
                            err_ind.avg_cell_ref_err += cell_err
                            cell_nref += 1
                            if kwargs["cell_ref_form"] == "hp":
                                err_ind.cols[col_key].cells[cell_key].ref_form = hp_steer_cell(mesh, proj, col_key, cell_key)
                        else:
                            err_ind.cols[col_key].cells[cell_key].ref_form = None
                            
            if kwargs["ref_col"]:
                err_ind.col_max_err = min(col_err, err_ind.col_max_err)
                err_ind.cols[col_key].err = col_err
                if col_err < kwargs["col_ref_tol"]:
                    err_ind.avg_col_ref_err += col_err
                    col_nref += 1
                    if kwargs["col_ref_form"] == "hp":
                        err_ind.cols[col_key].ref_form = hp_steer_col(mesh, proj, col_key)
                else:
                    err_ind.cols[col_key].ref_form = None
    #if kwargs["ref_col"]:
    #    if col_nref > 0.:
    #        err_ind.avg_col_ref_err /= col_nref
        
    #if kwargs["ref_cell"]:
    #    if cell_nref > 0.:
    #        err_ind.avg_cell_ref_err /= cell_nref
        
    return err_ind

def nneg_err_spt(mesh, proj, **kwargs):
    """
    Refine all with a negative value
    """
    
    # If integral of proejction has values less than the (negative) tolerance,
    # refine
    default_kwargs = {"ref_col"      : True,
                      "col_ref_form" : "hp",
                      "col_ref_kind" : "spt",
                      "col_ref_tol"  : 0.0,
                      "ref_cell"      : False,
                      "cell_ref_form" : None,
                      "cell_ref_kind" : None,
                      "cell_ref_tol"  : None}
    kwargs = {**default_kwargs, **kwargs}
    
    err_ind = Error_Indicator(mesh, **kwargs)
    col_nref = 0
    
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            if kwargs["ref_col"]:
                col_err = 10**10
            # Loop through cells
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, th1] = cell.pos[:]
                    dth        = th1 - th0
                    [nth]      = cell.ndofs[:]
                    
                    [_, _, _, _, _, wth] = qd.quad_xyth(nnodes_th = nth)
                    wth  = wth.reshape([1, 1, nth])
                    
                    uh_cell      = proj.cols[col_key].cells[cell_key].vals[:,:,:]
                    uh_cell_intg = (dth / 2.) * np.sum(wth * uh_cell, axis = (2))
                    
                    if kwargs["ref_col"]:
                        col_err = min(np.amin(uh_cell_intg), col_err)
                        
            if kwargs["ref_col"]:
                err_ind.col_max_err = min(col_err, err_ind.col_max_err)
                err_ind.cols[col_key].err = col_err
                if col_err < kwargs["col_ref_tol"]:
                    err_ind.avg_col_ref_err += col_err
                    col_nref += 1
                    if kwargs["col_ref_form"] == "hp":
                        err_ind.cols[col_key].ref_form = hp_steer_col(mesh, proj, col_key)
                else:
                    err_ind.cols[col_key].ref_form = None
    #if kwargs["ref_col"]:
    #    if col_nref > 0.:
    #        err_ind.avg_col_ref_err /= col_nref
        
    return err_ind
