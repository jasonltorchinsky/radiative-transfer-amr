# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# Local Library Imports
import consts
from dg.projection import Projection, push_forward
import dg.quadrature as qd

# Relative Imports

def plot_th(proj: Projection, lims: list = [[],[]], file_path: str = None,
            **kwargs) -> list:
    default_kwargs: dict = {"cmap"  : "hot",
                            "scale" : "normal",
                            "show_mesh" : True}
    kwargs: dict = {**default_kwargs, **kwargs}
    
    fig, ax = plt.subplots()
    
    ## Set plot range
    [Lx, Ly] = proj.mesh.Ls[:]
    if not lims[0]:
        xlim: list = [0, Lx]
    else:
        xlim: list = lims[0]
    if not lims[1]:
        ylim: list = [0, Ly]
    else:
        ylim: list = lims[1]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Spatially integrate each cell
    [rmin, rmax] = [consts.INF, -consts.INF]
    col_items: list = sorted(proj.cols.items())
    intg_xy_cols: dict = {}
    for col_key, col in col_items:
        assert(col.is_lf)

        [x0, y0, x1, y1] = col.pos[:]
        [dx, dy] = [x1 - x0, y1 - y0]
        [nx, ny] = col.ndofs[:]
        [_, wx, _, wy, _, _] = qd.quad_xyth(nnodes_x = nx,
                                            nnodes_y = ny)
        wx: np.ndarray = wx.reshape([nx, 1, 1])
        wy: np.ndarray = wy.reshape([1, ny, 1])
        
        dcoeff: float = (dx * dy) / 4.
        
        th_fs: list = []
        intg_xy_cells: list = []
        
        cell_items: list = sorted(col.cells.items())
        for cell_key, cell in cell_items:
            assert(cell.is_lf)

            [th0, th1] = cell.pos[:]
            [nth] = cell.ndofs[:]
            [_, _, _, _, th_b, _] = qd.quad_xyth(nnodes_th = nth)
            th_f: np.ndarray = push_forward(th0, th1, th_b)
            
            cell_vals: np.ndarray = proj.cols[col_key].cells[cell_key].vals[:,:,:]
            intg_xy_cell: np.ndarray = dcoeff * np.sum(wx * wy * cell_vals, axis = (0, 1))
            intg_xy_cell: np.ndarray = intg_xy_cell.reshape([nth])
            
            rmin: float = min(rmin, np.amin(intg_xy_cell))
            rmax: float = max(rmax, np.amax(intg_xy_cell))
            
            th_fs += [th_f]
            intg_xy_cells += [intg_xy_cell]
                
        th_f: np.ndarray = np.concatenate(th_fs)
        intg_xy_col: np.ndarray = np.concatenate(intg_xy_cells)
        
        sort_mask: np.ndarray = np.argsort(th_f)
        
        th_f: np.ndarray = th_f[sort_mask]
        intg_xy_col: np.ndarray = intg_xy_col[sort_mask]
        
        intg_xy_cols[col_key] = [th_f, intg_xy_col]
    
    # Plot angular distribution for each column
    rects: list = []
    for col_key, col in col_items:
        assert(col.is_lf)
        [x0, y0, x1, y1] = col.pos[:]
        [dx, dy] = [x1 - x0, y1 - y0]
        
        sub_ax = ax.inset_axes([x0 / Lx, y0 / Ly, dx / Lx, dy / Ly],
                               projection = "polar")
        intg_xy_col: np.ndarray = intg_xy_cols[col_key]
        th: np.ndarray = intg_xy_col[0]
        rr: np.ndarray = intg_xy_col[1]
        
        sub_ax.plot(th, rr, color = "k")
        sub_ax.set_rmin(rmin)
        sub_ax.set_rmax(rmax)
        sub_ax.set_xticks([], [])
        sub_ax.set_yticks([], [])
        sub_ax.set_rgrids([0.33 * rmax, 0.67 * rmax], angle = 90)
        
        rect: Rectangle = Rectangle((x0, y0), dx, dy,
                         facecolor = "none",
                         edgecolor = "black")
        rects += [rect]
            
    rect_coll: PatchCollection = PatchCollection(rects, match_original = True)
    ax.add_collection(rect_coll)

    if file_path:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.tight_layout()
        plt.savefig(file_path, dpi = 300)
        plt.close(fig)

    return [fig, ax]
