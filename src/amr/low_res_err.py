from copy import deepcopy
import numpy as np
from time import perf_counter

from .Error_Indicator import Error_Indicator
from .hp_steer import hp_steer_col, hp_steer_cell

import dg.quadrature as qd
from rt import rtdg

from utils import print_msg

phi_projs = {}
psi_projs = {}
xsi_projs = {}

def low_res_err(mesh, proj, kappa, sigma, Phi, bcs_dirac, f, **kwargs):
    """
    Create a low-resolution version of the mesh, solve the problem there,
    project the solution from the low-resolution mesh onto the original
    mesh, and find max-norm error from there.
    """
    
    default_kwargs = {'solver'       : 'spsolve',
                      'precondition' : False,
                      'verbose'      : False,
                      'ref_col'      : True,
                      'col_ref_form' : 'hp',
                      'col_ref_kind' : 'spt',
                      'col_ref_tol'  : 0.85,
                      'ref_cell'      : True,
                      'cell_ref_form' : 'hp',
                      'cell_ref_kind' : 'ang',
                      'cell_ref_tol'  : 0.85}
    kwargs = {**default_kwargs, **kwargs}
    
    col_items = sorted(mesh.cols.items())
    
    # Track maximum error(s) to calculate hp-steering only where needed
    col_max_err  = 0.
    col_ref_tol  = kwargs['col_ref_tol']
    cell_max_err = 0.
    cell_ref_tol = kwargs['cell_ref_tol']

    # Solve the problem on a lower-resolution mesh
    sub = 2
    mesh_lr = deepcopy(mesh)
    col_lr_items = sorted(mesh_lr.cols.items())
    for col_lr_key, col_lr in col_lr_items:
        if col_lr.is_lf:
            col_lr.ndofs[0] = max(1, col_lr.ndofs[0] - 2)
            col_lr.ndofs[1] = max(1, col_lr.ndofs[1] - 2)
            
            cell_lr_items = sorted(col_lr.cells.items())
            for cell_lr_key, cell_lr in cell_lr_items:
                if cell_lr.is_lf:
                    cell_lr.ndofs[0] = max(1, cell_lr.ndofs[0] - 2)
                    
    [bcs, dirac] = bcs_dirac
    proj_lr = rtdg(mesh_lr, kappa, sigma, Phi, [bcs, dirac], f, **kwargs)
    
    err_ind = Error_Indicator(mesh, **kwargs)
    
    # Relative error is weighted by the maximal value of the numeric solution, uh
    max_uh = 0.

    # Track maximum error(s) to calculate hp-steering only where needed
    col_max_err  = 0.
    col_ref_tol  = kwargs['col_ref_tol']
    cell_max_err = 0.
    cell_ref_tol = kwargs['cell_ref_tol']
    
    # Get max_norm(u_hr - uh) by column
    if kwargs['verbose']:
        t0 = perf_counter()

    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            # Column information for quadrature
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy] = [x1 - x0, y1 - y0]
            
            [nx, ny] = col.ndofs[:]
            [nx_lr, ny_lr] = mesh_lr.cols[col_key].ndofs[:]
            
            [xxb,    _, yyb,    _, _, _] = qd.quad_xyth(nnodes_x = nx,
                                                        nnodes_y = ny)
            [xxb_lr, _, yyb_lr, _, _, _] = qd.quad_xyth(nnodes_x = nx_lr,
                                                        nnodes_y = ny_lr)
            
            # Store spatial projection matrices for later reuse
            if (nx_lr, nx) in phi_projs.keys():
                phi_proj = phi_projs[(nx_lr, nx)][:]
            else:
                phi_proj = np.zeros([nx_lr, nx])
                for ii in range(0, nx_lr):
                    for pp in range(0, nx):
                        phi_proj[ii, pp] =  qd.lag_eval(xxb_lr, ii, xxb[pp])
                phi_projs[(nx_lr, nx)] = phi_proj[:]
                        
            if (ny_lr, ny) in psi_projs.keys():
                psi_proj = psi_projs[(ny_lr, ny)][:]
            else:
                psi_proj = np.zeros([ny_lr, ny])
                for jj in range(0, ny_lr):
                    for qq in range(0, ny):
                        psi_proj[jj, qq] = qd.lag_eval(yyb_lr, jj, yyb[qq])
                psi_projs[(ny_lr, ny)] = psi_proj[:]
                
            # Loop trough cells to calculate error
            col_err = 0.
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, th1] = cell.pos[:]
                    dth = th1 - th0
                    
                    [nth]    = cell.ndofs[:]
                    [nth_lr] = mesh_lr.cols[col_key].cells[cell_key].ndofs[:]

                    [_, _, _, _, thb,    _] = qd.quad_xyth(nnodes_th = nth)
                    [_, _, _, _, thb_lr, _] = qd.quad_xyth(nnodes_th = nth_lr)
                    
                    uh_cell  = proj.cols[col_key].cells[cell_key].vals
                    ulr_cell = proj_lr.cols[col_key].cells[cell_key].vals
                    
                    uh_lr_cell = np.zeros_like(uh_cell)
                    
                    # Store angular porjection matrices for later reuse                    
                    if (nth_lr, nth) in xsi_projs.keys():
                        xsi_proj = xsi_projs[(nth_lr, nth)][:]
                    else:
                        xsi_proj = np.zeros([nth_lr, nth])
                        for aa in range(0, nth_lr):
                            for rr in range(0, nth):
                                xsi_proj[aa, rr] = qd.lag_eval(thb_lr, aa, thb[rr])
                        xsi_projs[(nth_lr, nth)] = xsi_proj[:]
                        
                    for pp in range(0, nx):
                        for qq in range(0, ny):
                            for rr in range(0, nth):
                                for ii in range(0, nx_lr):
                                    phi_ip = phi_proj[ii, pp]
                                    for jj in range(0, ny_lr):
                                        psi_jq = psi_proj[jj, qq]
                                        for aa in range(0, nth_lr):
                                            xsi_ar = xsi_proj[aa, rr]
                                            uh_lr_cell[pp, qq, rr] += ulr_cell[ii, jj, aa] \
                                                * phi_ip * psi_jq * xsi_ar
                                            
                    # Calculate error
                    cell_err = np.amax(np.abs(uh_cell - uh_lr_cell))
                    col_err  = max(col_err, cell_err)
                    max_uh   = max(max_uh, np.amax(np.abs(uh_cell)))
                    
                    if kwargs['ref_cell']:
                        err_ind.cols[col_key].cells[cell_key].err_ind = cell_err
                        cell_max_err = max(cell_max_err, cell_err)
                        
            if kwargs['ref_col']:
                err_ind.cols[col_key].err_ind = col_err
                col_max_err = max(col_max_err, col_err)
            
    # Weight errors to be relative, and calculate hp-steering criteria
    if kwargs['ref_col']:
        col_max_err  /= max_uh
        col_ref_thrsh = col_ref_tol * col_max_err
        err_ind.col_max_err = col_max_err

    if kwargs['ref_cell']:
        cell_max_err  /= max_uh
        cell_ref_thrsh = cell_ref_tol * cell_max_err
        err_ind.cell_max_err = cell_max_err
        
    # Weight to be relative error
    for col_key, col in col_items:
        if col.is_lf:
            if kwargs['ref_col']: # If we're refining columns
                err_ind.cols[col_key].err /= max_uh
                if err_ind.cols[col_key].err >= col_ref_thrsh: # Does this one need to be refined?
                    if err_ind.cols[col_key].ref_form == 'hp': # Does the form of refinement need to be chosen?
                        err_ind.cols[col_key].ref_form = hp_steer_col(mesh, proj, col_key)
                else: # Needn't be refined
                    err_ind.cols[col_key].ref_form = None
                        
                
            if kwargs['ref_cell']: # If we're refining cells
                cell_items = sorted(col.cells.items())
                for cell_key, cell in cell_items:
                    if cell.is_lf:
                        err_ind.cols[col_key].cells[cell_key].err /= max_uh
                        
                        if err_ind.cols[col_key].cells[cell_key].err >= cell_ref_thrsh: # Does this one need to be refined?
                            if err_ind.cols[col_key].cells[cell_key].ref_form == 'hp': # Does the form of refinement need to be chosen?
                                err_ind.cols[col_key].cells[cell_key].ref_form = \
                                    hp_steer_cell(mesh, proj, col_key, cell_key)
                        else: # Needn't be refined
                            err_ind.cols[col_key].cells[cell_key].ref_form = None
            
    if kwargs['verbose']:
        tf = perf_counter()
        msg = (
            'Low-Res. Error Indicator Construction Time: {:8.4f} [s]\n'.format(tf - t0)
            )
        print_msg(msg)

    return err_ind
