import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from matplotlib.patches import Rectangle, Wedge

from .plot_mesh import plot_mesh, get_prism
from .. import get_cell_nhbr_in_col

def plot_nhbrs(mesh, col_key_0, cell_key_0 = None, file_name = None, **kwargs):
    
    default_kwargs = {'label_cells' : False}
    kwargs = {**default_kwargs, **kwargs}
    
    
    # Get neighboring column keys
    col_0 = mesh.cols[col_key_0]
    if col_0.is_lf:
        col_nhbr_keys = list(set(chain.from_iterable(col_0.nhbr_keys)))

    if cell_key_0 is not None:
        cell_0 = col_0.cells[cell_key_0]
        if cell_0.is_lf:
            cell_nhbr_keys = {col_key_0 : cell_0.nhbr_keys[:]}
            for col_key_1 in col_nhbr_keys:
                if col_key_1 is not None:
                    cell_nhbr_keys[col_key_1] = \
                        get_cell_nhbr_in_col(mesh = mesh,
                                             col_key = col_key_0,
                                             cell_key = cell_key_0,
                                             nhbr_col_key = col_key_1)
    else:
        cell_nhbr_keys = {}
    
    fig, ax = plt.subplots()

    [Lx, Ly] = mesh.Ls[:]
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])

    colors = ['#000000', '#E69F00', '#56B4E9']

    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy] = [x1 - x0, y1 - y0]
            [cx, cy] = [(x0 + x1) / 2., (y0 + y1) / 2.]

            if (col_key == col_key_0) and (cell_key_0 is None):
                facecolor = colors[0]
            elif (col_key in col_nhbr_keys) and (cell_key_0 is None):
                facecolor = colors[1]
            else:
                facecolor = 'none'
            rect = Rectangle((x0, y0), dx, dy,
                             facecolor = facecolor,
                             edgecolor = 'black')
            ax.add_patch(rect)

            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, th1] = cell.pos[:]
                    [deg0, deg1] = [th0 * 180. / np.pi, th1 * 180. / np.pi]
                    
                    if (col_key == col_key_0) and (cell_key == cell_key_0):
                        facecolor = colors[0]
                    elif col_key in cell_nhbr_keys.keys():
                        if cell_key in cell_nhbr_keys[col_key]:
                            if col_key == col_key_0:
                                facecolor = colors[1]
                            else:
                                facecolor = colors[2]
                        else:
                            facecolor = 'none'
                    else:
                        facecolor = 'none'
                        
                    wed = Wedge((cx, cy), min(dx, dy)/2, deg0, deg1,
                                facecolor = facecolor,
                                edgecolor = 'black'
                                )
                    ax.add_patch(wed)
                            
    if file_name:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.savefig(file_name, dpi = 300, bbox_inches = 'tight')
        plt.close(fig)
                
    return ax
