# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge
from matplotlib.collections import PatchCollection

# Local Library Imports
import consts
from dg.projection import Projection
import dg.quadrature as qd

# Relative Imports

def plot_xyth(proj: Projection, file_path: str = None, **kwargs) -> list:
    default_kwargs: dict = {"lims" : [[],[]],
                            "cmap"  : "hot",
                            "scale" : "normal"}
    kwargs: dict = {**default_kwargs, **kwargs}
    
    fig, ax = plt.subplots()
    
    ## Set plot range
    [Lx, Ly] = proj.mesh.Ls[:]
    if not kwargs["lims"][0]:
        xlim: list = [0, Lx]
    else:
        xlim: list = kwargs["lims"][0]
    if not kwargs["lims"][1]:
        ylim: list = [0, Ly]
    else:
        ylim: list = kwargs["lims"][1]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Get colorbar min/max
    [vmin, vmax] = [consts.INF, -consts.INF]
    col_items: list = sorted(proj.cols.items())
    cell_means: dict = {}
    for col_key, col in col_items:
        assert(col.is_lf)

        [nx, ny] = col.ndofs[:]
        [x0, y0, x1, y1] = col.pos[:]
        [dx, dy] = [x1 - x0, y1 - y0]
        [_, wx, _, wy, _, _] = qd.quad_xyth(nnodes_x = nx, nnodes_y = ny)
        wx: np.ndarray = wx.reshape([nx, 1, 1])
        wy: np.ndarray = wy.reshape([1, ny, 1])
        
        cell_items: list = sorted(col.cells.items())
        for cell_key, cell in cell_items:
            assert(cell.is_lf)
            
            [nth]      = cell.ndofs[:]
            [th0, th1] = cell.pos[:]
            [dth]      = [th1 - th0]
            [_, _, _, _, _, wth] = qd.quad_xyth(nnodes_th = nth)
            wth: np.ndarray = wth.reshape([1, 1, nth])
            
            cell_vals: np.ndarray = proj.cols[col_key].cells[cell_key].vals[:,:,:]
            cell_intg: float = (dx * dy * dth / 8.) * np.sum(wx * wy * wth * cell_vals)
            cell_mean: float = (1. / (dx * dy * dth)) * cell_intg

            vmin: float = min(vmin, cell_mean)
            vmax: float = max(vmax, cell_mean)

            cell_means[(col_key, cell_key)] = cell_mean

    # Two colors scales: diff shows a difference, pos shows positive values
    scale: str = kwargs["scale"]
    if scale == "diff":
        v_bnd: float = max(np.abs(vmin), np.abs(vmax))
        vmin: float = -v_bnd
        vmax: float = v_bnd
    elif scale == "pos":
        vmin: float = 0.
    elif scale == "normal":
        pass
    
    ## Get colormap
    cmap = plt.get_cmap(kwargs["cmap"])

    wedges: list = []
    wedge_colors: list = []
    rects: list  = []
    for col_key, col in col_items:
        assert(col.is_lf)
        
        [x0, y0, x1, y1] = col.pos[:]
        [dx, dy] = [x1 - x0, y1 - y0]
        [cx, cy] = [(x0 + x1) / 2., (y0 + y1) / 2.]

        rect: Rectangle = Rectangle((x0, y0), dx, dy,
                                    facecolor = "none",
                                    edgecolor = "black")
        rects += [rect]

        cell_items: list = sorted(col.cells.items())
        for cell_key, cell in cell_items:
            assert(cell.is_lf)
                
            [th0, th1]   = cell.pos[:]
            [deg0, deg1] = [th0 * 180. / consts.PI, th1 * 180. / consts.PI]

            cell_mean: float = cell_means[(col_key, cell_key)]
            
            wedge: Wedge = Wedge((cx, cy), min(dx, dy)/2, deg0, deg1,
                                 facecolor = cmap(cell_mean),
                                 edgecolor = None)
            wedges += [wedge]

            wedge_color: float = cell_mean
            wedge_colors += [wedge_color]
                    
    rect_coll: PatchCollection = PatchCollection(rects, match_original = True)
    ax.add_collection(rect_coll)
    
    wedge_coll: PatchCollection = PatchCollection(wedges, edgecolor = None,
                                                  cmap = cmap)
    wedge_coll.set_array(wedge_colors)
    wedge_coll.set_clim([vmin, vmax])
    ax.add_collection(wedge_coll)
    
    fig.colorbar(wedge_coll, ax = ax)
    
    if file_path:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.tight_layout()
        plt.savefig(file_path, dpi = 300)
        plt.close(fig)

    return [fig, ax]