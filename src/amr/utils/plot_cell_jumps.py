import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys

sys.path.append('../..')
from dg.quadrature import quad_xyth, lag_eval
from dg.projection import push_forward, pull_back

def plot_cell_jumps(mesh, proj, file_name = None, **kwargs):

    # Get number of Columns in mesh to get number of colums, rows in subplots
    ncol = len(mesh.cols.values())
    [nrow, ncol] = get_closest_factors(ncol)
    
    fig, axs = plt.subplots(nrow, ncol, sharex = True, sharey = True)
    
    col_items = sorted(mesh.cols.items())
    ax_idx = 0
    for col_key, col in col_items:
        if col.is_lf:
            
            ax_col_idx = int(np.mod(ax_idx, ncol))
            ax_row_idx = int(np.floor(ax_idx / ncol))
            
            ax = axs[ax_row_idx, ax_col_idx]
            
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy] = [x1 - x0, y1 - y0]
            [ndof_x, ndof_y] = col.ndofs[:]
            
            [_, w_x, _, w_y, _, _] = quad_xyth(nnodes_x = ndof_x,
                                               nnodes_y = ndof_y)
            
            w_x = w_x.reshape(ndof_x, 1, 1)
            w_y = w_y.reshape(1, ndof_y, 1)

            dcoeff = dx * dy / 4.
            
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, th1] = cell.pos[:]
                    [ndof_th] = cell.ndofs[:]
                    
                    [_, _, _, _, thb, _] = quad_xyth(nnodes_th = ndof_th)
                    
                    thf = push_forward(th0, th1, thb)
                    
                    u_cell = proj.cols[col_key].cells[cell_key].vals
                    u_th_cell = np.sum(dcoeff * w_x * w_y * u_cell, axis = (0, 1))
                    
                    ax.plot(thf, u_th_cell,
                            color     = 'black',
                            linestyle = '-',
                            linewidth = 0.8)
                    
                    ax.axvline(x = th1,
                               color     = 'gray',
                               linestyle = '--',
                               linewidth = 0.2)           
            
            title_str = 'Column {}'.format(col_key)
            ax.set_title(title_str)

            nth_ticks = 9
            th_ticks = np.linspace(0, 2, nth_ticks) * np.pi
            th_tick_labels = [None] * nth_ticks
            for aa in range(0, nth_ticks):
                th_rad = th_ticks[aa] / np.pi
                th_tick_labels[aa] = '{:.2f}\u03C0'.format(th_rad)
            ax.set_xticks(th_ticks)
            ax.set_xticklabels(th_tick_labels)
            
            ax_idx += 1
            
    if file_name:
        width  = (ncol / nrow) * 12
        height = (nrow / ncol) * 12
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
