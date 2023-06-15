import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge
from matplotlib.collections import PatchCollection
import sys

sys.path.append('../..')
from dg.quadrature import quad_xyth, lag_eval
from dg.projection import push_forward, pull_back

def plot_error_indicator(mesh, err_ind, file_name = None, **kwargs):
    
    default_kwargs = {'angles' : [0, np.pi/2, np.pi, 3*np.pi/2],
                      'cmap' : 'Reds',
                      'name' : '',
                      'by_col' : err_ind.by_col,
                      'by_cell' : err_ind.by_cell}
    kwargs = {**default_kwargs, **kwargs}

    if err_ind.by_col and kwargs['by_col']:
        [fig, ax] = plot_error_indicator_by_column(mesh,
                                                   err_ind,
                                                   file_name = file_name,
                                                   **kwargs)
    elif err_ind.by_cell and kwargs['by_cell']:
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
    [vmin, vmax] = [0., -10.**10]
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
                               cmap = kwargs['cmap'],
                               edgecolors = 'black')

    title_str = kwargs['name'] + ' Error Indicator'
    ax.set_title(title_str)
                
    fig.colorbar(pc)
    
    if file_name:
        fig.set_size_inches(12, 12 * (Ly / Lx))
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)

    return [fig, ax]

def plot_error_indicator_by_cell(mesh, err_ind, file_name = None, **kwargs):
    
    default_kwargs = {}
    kwargs = {**default_kwargs, **kwargs}
    
    cmap = cm.get_cmap(kwargs['cmap'])
    
    fig, ax = plt.subplots()
    
    [Lx, Ly] = mesh.Ls[:]
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    
    # Get colorbar min/max
    [vmin, vmax] = [0., -10.**10]
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    vmin = min(vmin, err_ind.cols[col_key].cells[cell_key].err_ind)
                    vmax = max(vmax, err_ind.cols[col_key].cells[cell_key].err_ind)

    wedges = []
    wedge_colors = []
    rects = []
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy] = [x1 - x0, y1 - y0]
            [cx, cy] = [(x0 + x1) / 2., (y0 + y1) / 2.]
            
            rect = Rectangle((x0, y0), dx, dy,
                             facecolor = 'none',
                             edgecolor = 'black')
            rects += [rect]

            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, th1] = cell.pos[:]
                    [deg0, deg1] = [th0 * 180. / np.pi, th1 * 180. / np.pi]
                    
                    cell_err = err_ind.cols[col_key].cells[cell_key].err_ind
                    cell_err_norm = (1. / (vmax - vmin)) * (cell_err - vmin)
                    
                    wedge = Wedge((cx, cy), min(dx, dy)/2, deg0, deg1,
                                  facecolor = cmap(cell_err),
                                  edgecolor = 'black')
                    wedges += [wedge]
                    wedge_color = cell_err
                    wedge_colors += [wedge_color]
                    
    rect_coll = PatchCollection(rects, match_original = True)
    ax.add_collection(rect_coll)
    
    wedge_coll = PatchCollection(wedges, edgecolor = 'black', cmap = cmap)
    wedge_coll.set_array(wedge_colors)
    wedge_coll.set_clim([vmin, vmax])
    ax.add_collection(wedge_coll)
    
    fig.colorbar(wedge_coll, ax = ax)
    
    if file_name:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)

    return [fig, ax]
