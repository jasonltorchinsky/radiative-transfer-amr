import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import sys

sys.path.append("../..")
import dg.quadrature as qd
from dg.projection import push_forward

def plot_cell_jumps(mesh, err_ind, file_name = None, **kwargs):
    
    default_kwargs = {}
    kwargs = {**default_kwargs, **kwargs}
    
    fig, ax = plt.subplots()
    
    [Lx, Ly] = mesh.Ls[:]
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    
    # Get list of cell jumps in each column
    [rmin, rmax] = [0., 0.]
    col_cell_jumps = {}
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            th_mids = []
            cell_jumps = []
            
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, th1] = cell.pos[:]
                    th_mid     = (th1 + th0) / 2.
                    
                    cell_jump = err_ind.cols[col_key].cells[cell_key].err
                    
                    rmin = min(rmin, cell_jump)
                    rmax = max(rmax, cell_jump)
                    
                    th_mids += [th_mid]
                    cell_jumps += [cell_jump]
            
            sort_mask = np.argsort(th_mids)
            
            th_mids    = np.array(th_mids)[sort_mask]
            cell_jumps = np.array(cell_jumps)[sort_mask]
            
            col_cell_jumps[col_key] = [th_mids, cell_jumps]
            
    # Add rectangular patches for mesh
    rects = []
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy] = [x1 - x0, y1 - y0]
            
            rect = Rectangle((x0, y0), dx, dy,
                             facecolor = "none",
                             edgecolor = "black")
            rects += [rect]
            
    rect_coll = PatchCollection(rects, match_original = True)
    ax.add_collection(rect_coll)
    
    # Add polar plots of cell jumps
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy] = [x1 - x0, y1 - y0]
            
            sub_ax = ax.inset_axes([x0 / Lx, y0 / Ly, dx / Lx, dy / Ly],
                                   projection = "polar")
            col_cell_jump = col_cell_jumps[col_key]
            th = col_cell_jump[0]
            rr = col_cell_jump[1]
            sub_ax.scatter(th, rr, color = "k", s = 0.25)
            sub_ax.set_rmin(rmin)
            sub_ax.set_rmax(1.1 * rmax)
            sub_ax.set_xticks([], [])
            sub_ax.set_yticks([], [])
            sub_ax.set_rgrids([err_ind.cell_ref_tol * rmax], angle = 90)
            
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title_str = "Max : {:.3E}, Min : {:.3E}".format(rmax, rmin)
    ax.set_title(title_str)
    
    if file_name:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.tight_layout()
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)

    return [fig, ax]
