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
    
    col_items = sorted(mesh.cols.items())
    
    ny = 6
    
    ys = np.linspace(0., Ly, ny)
    [nrows, ncols] = get_closest_factors(ny)

    # Create theta array based on the level of refinement of the cells
    max_ndof_th = 0
    max_lv = 0
    for col_key, col in col_items:
        if col.is_lf:
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    max_lv = max(max_lv, cell.lv)
                    max_ndof_th = max(max_ndof_th, cell.ndofs[0])

    min_dth    = 2. * np.pi / (2. * max_lv)
    coarse_th  = np.arange(0, 2. * np.pi + min_dth, min_dth)
    ncoarse_th = np.size(coarse_th)
    th = np.empty(0)
    for aa in range(0, ncoarse_th - 1):
        th0 = coarse_th[aa]
        th1 = coarse_th[aa + 1]
        [_, _, _, _, thb, _] = quad_xyth(nnodes_th = max_ndof_th)

        thf = push_forward(th0, th1, thb)

        th = np.concatenate([th, thf])

    nth = np.size(th)
    u  = np.zeros([nth])
    
    fig, axs = plt.subplots(nrows, ncols, sharex = True, sharey = True)
    
    for y_idx in range(0, ny):
        y = ys[y_idx]
        ax_col_idx = int(np.mod(y_idx, ncols))
        ax_row_idx = int(np.floor(y_idx / ncols))
        
        ax = axs[ax_row_idx, ax_col_idx]
        
        ax.set_title('Vertical Position: {:.2f}'.format(y))
        
        for col_key, col in col_items:
            if col.is_lf:
                [x0, y0, x1, y1] = col.pos[:]
                if ((y0 <= y) and (y <= y1)):
                    [dx, dy]         = [x1 - x0, y1 - y0]
                    [ndof_x, ndof_y] = col.ndofs[:]
                    [xxb, w_x, yyb, _, _, _] = quad_xyth(nnodes_x = ndof_x,
                                                         nnodes_y = ndof_y)
                    
                    y_pb = pull_back(y0, y1, y)
                    
                    proj_col = proj.cols[col_key]
                    cell_items = sorted(col.cells.items())
                    for cell_key, cell in cell_items:
                        if cell.is_lf:
                            [th0, th1] = cell.pos[:]
                            for th_idx in range(0, nth):
                                th_star = th[th_idx]
                                if (th0 <= th_star) and (th_star <= th1):
                                    [ndof_th]  = cell.ndofs

                                    [_, _, _, _, thb, _] = quad_xyth(nnodes_th = ndof_th)
                                    
                                    th_starb = pull_back(th0, th1, th_star)
                                    
                                    proj_cell = proj_col.cells[cell_key]
                                    vals_xyth = proj_cell.vals
                                    
                                    for ii in range(0, ndof_x):
                                        wx_i = w_x[ii]
                                        for jj in range(0, ndof_y):
                                            psi_j = lag_eval(yyb, jj, y_pb)
                                            for aa in range(0, ndof_th):
                                                xsi_a = lag_eval(thb, aa, th_starb)
                                                u[th_idx] += (dx / 2.) * wx_i \
                                                    * vals_xyth[ii, jj, aa]  * psi_j * xsi_a

                            ax.axvline(th1,
                                       color = 'gray',
                                       linestyle = '--',
                                       linewidth = 0.2)
        
        ax.plot(th, u,
                color = 'black',
                linestyle = '-',
                linewidth = 1)

        # Get vertical limits
        [u_min, u_max] = [np.amin(u), np.amax(u)]
        
        for ax in axs.flatten():
            ax.set_xlim([0, Lth])
            ax.set_ylim([0.9 * u_min, 1.1 * u_max])

        nth_ticks = 9
        th_ticks = np.linspace(0, 2, nth_ticks) * np.pi
        th_tick_labels = [None] * nth_ticks
        for aa in range(0, nth_ticks):
            th_rad = th_ticks[aa] / np.pi
            th_tick_labels[aa] = '{:.2f}\u03C0'.format(th_rad)
        ax.set_xticks(th_ticks)
        ax.set_xticklabels(th_tick_labels)
    
    if file_name:
        width  = (ncols / nrows) * 12
        height = (nrows / ncols) * 12
        fig.set_size_inches(width, height)
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)
        
    return [fig, ax]

def get_closest_factors(x):
    '''
    Gets the factors of x that are closest to the square root.
    '''

    a = int(np.floor(np.sqrt(x)))
    while ((x / a) - np.floor(x / a) > 10**(-3)):
        a = int(a - 1)

    return [int(a), int(x / a)]
