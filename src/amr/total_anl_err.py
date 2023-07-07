import numpy as np
from scipy.integrate import nquad

import dg.quadrature as qd
from dg.projection import push_forward

intg_u2 = None

def total_anl_err(mesh, proj, anl_sol, **kwargs):
    """
    Calculate the L2-error by cell (and column), weighted to be the relative error.
    """
    
    default_kwargs = {}
    kwargs = {**default_kwargs, **kwargs}
    
    col_items = sorted(mesh.cols.items())
    
    # Integrate th analytic solution and the square of the analytic solution here
    global intg_u2
    if intg_u2 is None:
        print('gotta get intg_u2')
        [Lx, Ly] = mesh.Ls[:]
        [intg_u2, _] = nquad(lambda x, y, th: (anl_sol(x, y, th))**2,
                             [[0, Lx], [0, Ly], [0, 2. * np.pi]])
    err = 0.
    # Calculate the errors
    for col_key, col in col_items:
        if col.is_lf:
            # Column information for quadrature
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy]         = [x1 - x0, y1 - y0]
            [ndof_x, ndof_y] = col.ndofs[:]
            
            [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                    nnodes_y = ndof_y)
            
            xxf = push_forward(x0, x1, xxb).reshape(ndof_x, 1, 1)
            wx  = wx.reshape(ndof_x, 1, 1)
            yyf = push_forward(y0, y1, yyb).reshape(1, ndof_y, 1)
            wy  = wy.reshape(1, ndof_y, 1)
            
            # Loop through cells to calculate error
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    # Cell information for quadrature
                    [th0, th1] = cell.pos[:]
                    dth = th1 - th0
                    [ndof_th]  = cell.ndofs[:]
                    
                    [_, _, _, _, thb, wth] = qd.quad_xyth(nnodes_th = ndof_th)
                    
                    thf = push_forward(th0, th1, thb).reshape(1, 1, ndof_th)
                    wth = wth.reshape(1, 1, ndof_th)
                    
                    # Calculate error
                    uh_cell = proj.cols[col_key].cells[cell_key].vals
                    u_cell  = anl_sol(xxf, yyf, thf)
                    
                    err += (dx * dy * dth / 8.) \
                        * np.sum(wx * wy * wth * (u_cell - uh_cell)**2)
                    
    return err/intg_u2
