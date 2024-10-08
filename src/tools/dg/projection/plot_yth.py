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

def plot_yth(proj: Projection, file_path: str = None, **kwargs) -> list:
    default_kwargs: dict = {"lims" : [[],[]],
                            "cmap"  : "hot",
                            "scale" : "normal",
                            "show_mesh" : True}
    kwargs: dict = {**default_kwargs, **kwargs}
    
    fig, ax = plt.subplots()
    
    ## Set plot range
    [Lth, Ly] = [2. * consts.PI, proj.mesh.Ls[1]]
    if not kwargs["lims"][0]:
        thlim: list = [0, Lth]
    else:
        thlim: list = kwargs["lims"][0]
    if not kwargs["lims"][1]:
        ylim: list = [0, Ly]
    else:
        ylim: list = kwargs["lims"][1]
    ax.set_xlim(thlim)
    ax.set_ylim(ylim)

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("y")

    # Integrate each cell in x, and get the bounds for the colorbar
    [vmin, vmax] = [consts.INF, -consts.INF]
    col_items: list = sorted(proj.cols.items())
    cell_intg_ys: dict = {}
    for col_key, col in col_items:
        assert(col.is_lf)

        [nx, _] = col.ndofs[:]
        [x0, _, x1, _] = col.pos[:]
        dx: float = x1 - x0
            
        [_, wx, _, _, _, _] = qd.quad_xyth(nnodes_x = nx)
        wx: np.ndarray = wx.reshape([nx, 1, 1])
            
        cell_items = sorted(col.cells.items())
        for cell_key, cell in cell_items:
            assert(cell.is_lf)

            [ndof_th] = cell.ndofs[:]
            cell_vals: np.ndarray = proj.cols[col_key].cells[cell_key].vals[:,:,:]
            
            cell_intg_x: np.ndarray = (dx / 2.) * np.sum(wx * cell_vals, axis = 0)
                
            cell_intg_ys[(col_key, cell_key)] = cell_intg_x
            vmin: float = min(np.min(cell_intg_x), vmin)
            vmax: float = max(np.max(cell_intg_x), vmax)
                    
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

        [_, y0, _, y1] = col.pos[:]
        [_, ny] = col.ndofs[:]
        
        if kwargs["show_mesh"]:
            dy: float = y1 - y0

        [_, _, yyb, _, _, _] = qd.quad_xyth(nnodes_y = ny)
        
        yyf: np.ndarray = push_forward(y0, y1, yyb)
        
        cell_items: list = sorted(col.cells.items())
        for cell_key, cell in cell_items:
            assert(cell.is_lf)
                
            [th0, th1] = cell.pos[:]
            
            [ndof_th]  = cell.ndofs[:]
            [_, _, _, _, thb, _] = qd.quad_xyth(nnodes_th = ndof_th)
            
            thf: np.ndarray = push_forward(th0, th1, thb)
            vals: np.ndarray = cell_intg_ys[(col_key, cell_key)]
            
            pc = ax.pcolormesh(thf, yyf, vals,
                               cmap = cmap,
                               vmin = vmin, vmax = vmax,
                               shading = "gouraud")
            
            if kwargs["show_mesh"]:
                dth: float = th1 - th0    
                rect: Rectangle = Rectangle((th0, y0), dth, dy,
                                 facecolor = "none",
                                 edgecolor = "black")
                rects += [rect]
    
    if kwargs["show_mesh"]:
        rect_coll: PatchCollection = PatchCollection(rects, match_original = True)
        ax.add_collection(rect_coll)

    fig.colorbar(pc, ax = ax)
    
    ## Set theta-dimension ticks
    nth_ticks: int = 9
    th_ticks: np.ndarray = np.linspace(0, 2, nth_ticks) * consts.PI
    th_tick_labels: list = [None] * nth_ticks
    for aa in range(0, nth_ticks):
        th_rad: float = th_ticks[aa] / consts.PI
        th_tick_labels[aa] = "{:.2f}\u03C0".format(th_rad)
    ax.set_xticks(th_ticks)
    ax.set_xticklabels(th_tick_labels)
    
    if file_path:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lth))
        plt.tight_layout()
        plt.savefig(file_path, dpi = 300)
        plt.close(fig)
        
    return [fig, ax]
