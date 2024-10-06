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

def plot_xy(proj: Projection, lims: list = [[],[]], file_path: str = None,
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

    # Get colorbar min/max
    [vmin, vmax] = [consts.INF, -consts.INF]
    col_items: list = sorted(proj.cols.items())
    col_intg_ths: dict = {}
    for col_key, col in col_items:
        assert(col.is_lf)

        [nx, ny] = col.ndofs[:]
        col_intg_th: np.ndarray = np.zeros([nx, ny])
        
        cell_items: list = sorted(col.cells.items())
        for cell_key, cell in cell_items:
            assert(cell.is_lf)

            [nth]    = cell.ndofs[:]
            [th_0, th_1] = cell.pos[:]
            dth: float   = th_1 - th_0
            [_, _, _, _, _, wth] = qd.quad_xyth(nnodes_th = nth)
            wth: np.ndarray = wth.reshape([1, 1, nth])

            cell_vals: np.ndarray = proj.cols[col_key].cells[cell_key].vals[:,:,:]
            
            col_intg_th += (dth / 2.) * np.sum(wth * cell_vals, axis = 2)
                    
        col_intg_ths[col_key] = col_intg_th
        vmin: float = min(np.min(col_intg_th), vmin)
        vmax: float = max(np.max(col_intg_th), vmax)
            
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

    rects: list = []
    for col_key, col in col_items:
        assert(col.is_lf)

        [x0, y0, x1, y1] = col.pos[:]
        [nx, ny] = col.ndofs[:]
        
        [xxb, _, yyb, _, _, _] = qd.quad_xyth(nnodes_x = nx,
                                              nnodes_y = ny)
        
        xxf: np.ndarray = push_forward(x0, x1, xxb)
        yyf: np.ndarray = push_forward(y0, y1, yyb)
        
        vals: np.ndarray = col_intg_ths[col_key]
        
        pc = ax.pcolormesh(xxf, yyf, vals.transpose(),
                           cmap = cmap,
                           vmin = vmin, vmax = vmax,
                           shading = "gouraud")
        
        if kwargs["show_mesh"]:
            [dx, dy] = [x1 - x0, y1 - y0]
            rect: Rectangle = Rectangle((x0, y0), dx, dy,
                                        facecolor = "none",
                                        edgecolor = "black")
            rects += [rect]
    
    if kwargs["show_mesh"]:
        rect_coll: PatchCollection = PatchCollection(rects, match_original = True)
        ax.add_collection(rect_coll)

    fig.colorbar(pc, ax = ax)
    
    if file_path:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.tight_layout()
        plt.savefig(file_path, dpi = 300)
        plt.close(fig)
        
    return [fig, ax]
