import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge
from matplotlib.collections import PatchCollection
import sys

sys.path.append('../..')
import dg.quadrature as qd
from dg.projection import push_forward, pull_back

def plot_xyth(mesh, proj, file_name = None, **kwargs):
    
    default_kwargs = {'cmap'  : 'hot',
                      'scale' : 'normal'}
    kwargs = {**default_kwargs, **kwargs}
    
    cmap = cm.get_cmap(kwargs['cmap'])
    
    fig, ax = plt.subplots()
    
    [Lx, Ly] = mesh.Ls[:]
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    
    # Get colorbar min/max
    [vmin, vmax] = [10.**10, -10.**10]
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    cell_mean = np.mean(proj.cols[col_key].cells[cell_key].vals)
                    vmin = min(vmin, cell_mean)
                    vmax = max(vmax, cell_mean)

    scale = kwargs['scale']
    if scale == 'diff':
        v_bnd = max(np.abs(vmin), np.abs(vmax))
        vmin = -v_bnd
        vmax = v_bnd
    elif scale == 'pos':
        vmin = 0.
    # Default to a normal color scale

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
                    
                    cell_mean = np.mean(proj.cols[col_key].cells[cell_key].vals)
                    
                    wedge = Wedge((cx, cy), min(dx, dy)/2, deg0, deg1,
                                  facecolor = cmap(cell_mean),
                                  edgecolor = 'black')
                    wedges += [wedge]
                    wedge_color = cell_mean
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
