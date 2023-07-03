import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import sys

sys.path.append('../..')
import dg.quadrature as qd
from dg.projection import push_forward

def plot_th(mesh, proj, file_name = None, **kwargs):
    
    default_kwargs = {}
    kwargs = {**default_kwargs, **kwargs}
    
    fig, ax = plt.subplots()
    
    [Lx, Ly] = mesh.Ls[:]
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    
    # Spatially integrate each cell
    [rmin, rmax] = [0., 0.]
    intg_xy_cols = {}
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy] = [x1 - x0, y1 - y0]
            [nx, ny] = col.ndofs[:]
            [_, wx, _, wy, _, _] = qd.quad_xyth(nnodes_x = nx,
                                                nnodes_y = ny)
            wx = wx.reshape([nx, 1, 1])
            wy = wy.reshape([1, ny, 1])
            
            dcoeff = (dx * dy) / 4.
            
            th_fs = []
            intg_xy_cells = []
            
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, th1] = cell.pos[:]
                    [nth] = cell.ndofs[:]
                    [_, _, _, _, th_b, _] = qd.quad_xyth(nnodes_th = nth)
                    th_f = push_forward(th0, th1, th_b)
                    
                    cell_vals = proj.cols[col_key].cells[cell_key].vals[:]
                    intg_xy_cell = dcoeff * np.sum(wx * wy * cell_vals, axis = (0, 1))
                    intg_xy_cell = intg_xy_cell.reshape([nth])
                    
                    rmin = min(rmin, np.amin(intg_xy_cell))
                    rmax = max(rmax, np.amax(intg_xy_cell))
                    
                    th_fs += [th_f]
                    intg_xy_cells += [intg_xy_cell]
                    
            th_f = np.concatenate(th_fs)
            intg_xy_col = np.concatenate(intg_xy_cells)
            
            sort_mask = np.argsort(th_f)
            
            th_f = th_f[sort_mask]
            intg_xy_col = intg_xy_col[sort_mask]
            
            intg_xy_cols[col_key] = [th_f, intg_xy_col]
            
    # Add rectangular patches for mesh
    rects = []
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy] = [x1 - x0, y1 - y0]
            
            rect = Rectangle((x0, y0), dx, dy,
                             facecolor = 'none',
                             edgecolor = 'black')
            rects += [rect]
            
    rect_coll = PatchCollection(rects, match_original = True)
    ax.add_collection(rect_coll)
    
    # Add polar plots
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy] = [x1 - x0, y1 - y0]
            
            sub_ax = ax.inset_axes([x0 / Lx, y0 / Ly, dx / Lx, dy / Ly],
                                   projection = 'polar')
            intg_xy_col = intg_xy_cols[col_key]
            th = intg_xy_col[0]
            rr = intg_xy_col[1]
            sub_ax.plot(th, rr, color = 'k')
            sub_ax.set_rmin(rmin)
            sub_ax.set_rmax(rmax)
            sub_ax.set_xticks([], [])
            sub_ax.set_yticks([], [])
            sub_ax.set_rgrids([0.33 * rmax, 0.67 * rmax], angle = 90)
            
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    if file_name:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.tight_layout()
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)

    return [fig, ax]
