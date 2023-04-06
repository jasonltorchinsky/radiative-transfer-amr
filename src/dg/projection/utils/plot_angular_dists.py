import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

sys.path.append('../..')
from dg.quadrature import quad_xyth, lag_eval
from dg.projection import push_forward, pull_back

def plot_angular_dists(mesh, proj, file_name = None, **kwargs):

    if not mesh.has_th:
        return None
    
    [Lx, Ly] = mesh.Ls[:]
    Lth = 2. * np.pi

    ny = 6
    
    x  = Lx / 3.
    ys = np.linspace(0., Ly, ny)
    [nrows, ncols] = get_closest_factors(ny)

    th = np.empty(0)
    u  = np.empty(0)
    
    fig, axs = plt.subplots(nrows, ncols, sharex = True, sharey = True)
    # Get vertical limits
    [u_min, u_max] = [10.**10, -10.**10]
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            proj_col = proj.cols[col_key]
            cell_items = sorted(proj_col.cells.items())
            for cell_key, cell in cell_items:
                u_min = min(u_min, np.amin(cell.vals))
                u_max = max(u_max, np.amax(cell.vals))

    for ax in axs.flatten():
        ax.set_xlim([0, Lth])
        ax.set_ylim([0.9 * u_min, 1.1 * u_max])
    
    for y_idx in range(0, ny):
        y = ys[y_idx]
        ax_col_idx = int(np.mod(y_idx, ncols))
        ax_row_idx = int(np.floor(y_idx / ncols))
        
        ax = axs[ax_row_idx, ax_col_idx]
        
        ax.set_title('Vertical Position: {:.2f}'.format(y))
        
        for col_key, col in col_items:
            if col.is_lf:
                [x0, y0, x1, y1] = col.pos[:]
                if ((x0 <= x) and (x <= x1) and
                    (y0 <= y) and (y <= y1)):
                    [ndof_x, ndof_y] = col.ndofs[:]
                    [xxb, _, yyb, _, _, _] = quad_xyth(nnodes_x = ndof_x,
                                                       nnodes_y = ndof_y)
                    
                    x_pb = pull_back(x0, x1, x)
                    y_pb = pull_back(y0, y1, y)
                    
                    proj_col = proj.cols[col_key]
                    cell_items = sorted(col.cells.items())
                    for cell_key, cell in cell_items:
                        if cell.is_lf:
                            [th0, th1] = cell.pos[:]
                            [ndof_th]  = cell.ndofs
                            
                            [_, _, _, _, thb, _] = quad_xyth(nnodes_th = ndof_th)
                            thf = push_forward(th0, th1, thb)
                            
                            proj_cell = proj_col.cells[cell_key]
                            vals_xyth = proj_cell.vals
                            u_cell = np.zeros([ndof_th])
                            
                            for ii in range(0, ndof_x):
                                phi_i = lag_eval(xxb, ii, x_pb)
                                for jj in range(0, ndof_y):
                                    psi_j = lag_eval(yyb, jj, y_pb)
                                    for aa in range(0, ndof_th):
                                        u_cell[aa] += vals_xyth[ii, jj, aa] \
                                            * phi_i * psi_j
                            
                            th = np.concatenate([th, thf])
                            u = np.concatenate([u, u_cell])

                            ax.axvline(thf[-1],
                                       color = 'gray',
                                       linestyle = '--',
                                       linewidth = 0.2)
        
        sort = np.argsort(th)
        th = th[sort]
        u = u[sort]
        ax.plot(th, u,
                color = 'black',
                linestyle = '-',
                linewidth = 1)
    
    if file_name:
        width  = (ncols / nrows) * 12
        height = (nrows / ncols) * 12
        fig.set_size_inches(width, height)
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)
        
    return [fig, ax]

"""
    [Lx, Ly] = mesh.Ls[:]
    default_kwargs = {'xs' : np.linspace(0.0, Lx, 6),
                      'cmap' : 'hot'}
    kwargs = {**default_kwargs, **kwargs}
    
    xs  = kwargs['xs']
    nxs = np.size(xs)

    Lth = 2. * np.pi

    # Number of sampling points for each cell
    ny = 16
    nth = 16

    # Set up the subplots
    [nrows, ncols] = get_closest_factors(nxs)

    fig, axs = plt.subplots(nrows, ncols, sharex = True, sharey = True)
    for ax in axs.flatten():
        ax.set_xlim([0, Lth])
        ax.set_ylim([0, Ly])
    
    # Get colorbar min/max
    [vmin, vmax] = [10.**10, -10.**10]
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:  
        # There should only be one cell, so this should be okay
        proj_col = proj.cols[col_key]
        cell_items = sorted(proj_col.cells.items())
        for cell_key, cell in cell_items:
            vmin = min(vmin, np.amin(cell.vals))
            vmax = max(vmax, np.amax(cell.vals))

    for x_idx in range(0, nxs):
        x = xs[x_idx]
        ax_x_idx = int(np.mod(x_idx, ncols))
        ax_y_idx = int(np.floor(x_idx / ncols))

        ax = axs[ax_y_idx, ax_x_idx]

        # Title
        ax.set_title('Horizontal Position: {:.2f}'.format(x))
        
        for col_key, col in col_items:
            if col.is_lf:# Plot column
                [x0, y0, x1, y1] = col.pos[:]
                [ndof_x, ndof_y] = col.ndofs[:]
                
                if (x0 <= x) and (x <= x1):
                    yy = np.linspace(y0, y1, ny)
                    [xxb, _, yyb, _, _, _] = quad_xyth(nnodes_x = ndof_x,
                                                       nnodes_y = ndof_y)
                    x_pb = pull_back(x0, x1, x)
                    yy_pb = pull_back(y0, y1, yy)
                    
                    proj_col = proj.cols[col_key]
                    cell_items = sorted(proj_col.cells.items())
                    for cell_key, cell in cell_items:
                        [th0, th1] = cell.pos[:]
                        [ndof_th]  = cell.ndofs
                        
                        th = np.linspace(th0, th1, nth)
                        
                        [_, _, _, _, thb, _] = quad_xyth(nnodes_th = ndof_th)
                        th_pb = pull_back(th0, th1, th)
                        
                        vals_xyth = cell.vals[:,:,:]
                        vals_yth = np.zeros([ny, nth])
                        for ii in range(0, ndof_x):
                            phi_i = lag_eval(xxb, ii, x_pb)
                            for y_idx in range(0, ny):
                                y_pt = yy_pb[y_idx]
                                for jj in range(0, ndof_y):
                                    psi_j = lag_eval(yyb, jj, y_pt)
                                    for th_idx in range(0, ndof_th):
                                        th_pt = th_pb[th_idx]
                                        for aa in range(0, ndof_th):
                                            xsi_a = lag_eval(thb, aa, th_pt)

                                            vals_yth[y_idx, th_idx] += vals_xyth[ii, jj, aa] \
                                                * phi_i * psi_j * xsi_a
                                    
                        pc = ax.pcolormesh(th, yy, vals_yth.transpose(),
                                           shading = 'gouraud',
                                           vmin = vmin, vmax = vmax,
                                           cmap = kwargs['cmap'])
    
    for x_idx in range(0, nxs):
        x = xs[x_idx]
        ax_x_idx = int(np.mod(x_idx, ncols))
        ax_y_idx = int(np.floor(x_idx / ncols))

        ax = axs[ax_y_idx, ax_x_idx]
        
        for col_key, col in col_items:
            if col.is_lf:
                [x0, y0, x1, y1] = col.pos[:]
                [dx, dy] = [x1 - x0, y1 - y0]
                if (x0 <= x) and (x <= x1):
                    proj_col = proj.cols[col_key]
                    cell_items = sorted(proj_col.cells.items())
                    for cell_key, cell in cell_items:
                        [th0, th1] = cell.pos[:]
                        dth = th1 - th0
                        
                        rect = Rectangle((th0, y0), dth, dy, fill = False)
                        ax.add_patch(rect)
            
    fig.colorbar(pc, ax = axs, location = 'right')
    
    if file_name:
        width  = (ncols / nrows) * 12
        height = (nrows / ncols) * 12
        fig.set_size_inches(width, height)
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)
        
    return [fig, ax]
"""

def get_closest_factors(x):
    '''
    Gets the factors of x that are closest to the square root.
    '''

    a = int(np.floor(np.sqrt(x)))
    while ((x / a) - np.floor(x / a) > 10**(-3)):
        a = int(a - 1)

    return [int(a), int(x / a)]
