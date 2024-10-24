# Standard Library Imports

# Third-Party Library Imports
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge
from matplotlib.collections import PatchCollection

# Local Library Imports
import consts
from amr.error_indicator import Error_Indicator

# Relative Imports

def plot_error_indicator(err_ind: Error_Indicator, file_path: str = None,
                         **kwargs):
    default_kwargs: dict = {"lims" : [[],[]],
                            "name" : "",
                            "cmap" : "Reds",
                            "kind" : "all"}
    kwargs: dict = {**default_kwargs, **kwargs}

    fig, ax = plt.subplots()
    
    ## Set plot range
    [Lx, Ly] = err_ind.proj.mesh.Ls[:]
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


    ## Set axis labels and title
    title: str = ( "Total Error: {:4.4E}\n".format(err_ind.error) +
                   "Error to Resolve: {:4.4E}".format(err_ind.error_to_resolve) )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    
    # Get colorbar min/max
    vmin: float = 0.
    vmax: float = max(err_ind.col_max_error, err_ind.cell_max_error)
    
    ## Get colormap
    cmap = plt.get_cmap(kwargs["cmap"])

    wedges: list = []
    wedge_colors: list = []
    rects: list = []
    rect_colors: list = []
    col_items: list = sorted(err_ind.proj.cols.items())
    for col_key, col in col_items:
        assert(col.is_lf)
        
        [x0, y0, x1, y1] = col.pos[:]
        [dx, dy] = [x1 - x0, y1 - y0]
        [cx, cy] = [(x0 + x1) / 2., (y0 + y1) / 2.]
        
        col_err: float = err_ind.cols[col_key].error

        if kwargs["kind"] in ["spt", "all"]:
            facecolor: str = cmap(col_err)
        else:
            faceolor: str = "none"

        rect: Rectangle = Rectangle((x0, y0), dx, dy,
                                    facecolor = facecolor,
                                    edgecolor = "black")
        rects += [rect]

        rect_color: float = col_err
        rect_colors += [rect_color]

        cell_items: list = sorted(col.cells.items())
        for cell_key, cell in cell_items:
            assert(cell.is_lf)
                
            [th0, th1]   = cell.pos[:]
            [deg0, deg1] = [th0 * 180. / consts.PI, th1 * 180. / consts.PI]

            cell_err: float = err_ind.cols[col_key].cells[cell_key].error
            
            if kwargs["kind"] in ["ang", "all"]:
                facecolor: str = cmap(cell_err)
            else:
                faceolor: str = "none"

            wedge: Wedge = Wedge((cx, cy), min(dx, dy)/2, deg0, deg1,
                                 facecolor = facecolor,
                                 edgecolor = "black")
            wedges += [wedge]

            wedge_color: float = cell_err
            wedge_colors += [wedge_color]
                    
    rect_coll: PatchCollection = PatchCollection(rects, match_original = True,
                                                 cmap = cmap)
    rect_coll.set_array(rect_colors)
    rect_coll.set_clim([vmin, vmax])
    ax.add_collection(rect_coll)
    
    wedge_coll: PatchCollection = PatchCollection(wedges, match_original = True,
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