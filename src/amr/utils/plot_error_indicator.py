import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys

sys.path.append('../..')
from dg.quadrature import quad_xyth, lag_eval
from dg.projection import push_forward, pull_back

def plot_error_indicator(mesh, err_ind, file_name = None, **kwargs):
    
    default_kwargs = {'angles' : [0, np.pi/2, np.pi, 3*np.pi/2],
                      'cmap' : 'Reds',
                      'name' : ''}
    kwargs = {**default_kwargs, **kwargs}

    if err_ind.by_col:
        [fig, ax] = plot_error_indicator_by_column(mesh,
                                                   err_ind,
                                                   file_name = file_name,
                                                   **kwargs)
    if err_ind.by_cell:
        [fig, ax] = plot_error_indicator_by_cell(mesh,
                                                 err_ind,
                                                 file_name = file_name,
                                                 **kwargs)
        
    return [fig, ax]

def plot_error_indicator_by_column(mesh, err_ind, file_name = None, **kwargs):

    default_kwargs = {}
    kwargs = {**default_kwargs, **kwargs}
    
    [Lx, Ly] = mesh.Ls[:]
    
    fig, ax = plt.subplots()
    
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])

    # Get colorbar min/max
    [vmin, vmax] = [10.**10, -10.**10]
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            vmin = min(vmin, err_ind.cols[col_key].err_ind)
            vmax = max(vmax, err_ind.cols[col_key].err_ind)
    
    for col_key, col in col_items:
        if col.is_lf:
            # Plot column err indicator
            [x0, y0, x1, y1] = col.pos
            [ndof_x, ndof_y] = col.ndofs
            col_err_ind = err_ind.cols[col_key].err_ind
            
            xx = np.asarray([x0, x1])
            yy = np.asarray([y0, y1])
            pc = ax.pcolormesh(xx, yy, [[col_err_ind]], shading = 'flat',
                               vmin = vmin, vmax = vmax,
                               cmap = kwargs['cmap'])
        

    for col_key, col in col_items:
        # Plot column
        [x0, y0, x1, y1] = col.pos
        [dx, dy] = [x1 - x0, y1 - y0]
        
        rect = Rectangle((x0, y0), dx, dy, fill = False)
        ax.add_patch(rect)

    title_str = kwargs['name'] + ' Error Indicator'
    ax.set_title(title_str)
                
    fig.colorbar(pc)
    
    if file_name:
        fig.set_size_inches(12, 12 * (Ly / Lx))
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)

    return [fig, ax]

def plot_error_indicator_by_cell(mesh, err_ind, file_name = None, **kwargs):

    default_kwargs = {'angles' : [0, np.pi/2, np.pi, 3*np.pi/2]}
    kwargs = {**default_kwargs, **kwargs}
    
    [Lx, Ly] = mesh.Ls[:]
    Lz       = 2 * np.pi
    angles   = kwargs['angles']
    nangles  = np.shape(angles)[0]

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
            
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, th1] = cell.pos[:]
                    
                    cell_err_ind = err_ind.cols[col_key].cells[cell_key].err_ind
                    
                    th = np.asarray([th0, th1])
                    errs = cell_err_ind * np.ones_like(th)
                    
                    ax.plot(th, errs,
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
