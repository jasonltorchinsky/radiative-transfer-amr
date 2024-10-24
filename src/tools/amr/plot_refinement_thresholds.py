# Standard Library Imports

# Third-Party Library Imports
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle, Wedge
from matplotlib.collections import PatchCollection

# Local Library Imports
import consts
from amr.error_indicator import Error_Indicator

# Relative Imports

def plot_refinement_thresholds(err_ind: Error_Indicator, file_path: str = None,
                               **kwargs):
    default_kwargs: dict = {"lims" : [[],[]],
                            "name" : "",
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

    wedges: list = []
    rect_elems: list = []
    rect_thrshs: list = []
    rect_errs: list = []
    col_items: list = sorted(err_ind.proj.cols.items())
    for col_key, col in col_items:
        assert(col.is_lf)
        
        [x0, y0, x1, y1] = col.pos[:]
        [dx, dy] = [x1 - x0, y1 - y0]
        [cx, cy] = [(x0 + x1) / 2., (y0 + y1) / 2.]
        
        col_err: float = err_ind.cols[col_key].error

        ## Plot the element boundary
        rect_elem: Rectangle = Rectangle((x0, y0), dx, dy,
                                         facecolor = "none",
                                         edgecolor = "black")
        
        rect_elems += [rect_elem]

        ## Plot the spatial refinement threshold and spatial error
        if kwargs["kind"] in ["spt", "all"]:
            ## Plot the spatial refinement threshold
            anchor_x: float = x0 + (dx / 2.) * (1. - err_ind.spt_ref_tol)
            anchor_y: float = y0 + (dy / 2.) * (1. - err_ind.spt_ref_tol)
            width: float = dx * err_ind.spt_ref_tol
            height: float = dy * err_ind.spt_ref_tol
            rect_thrsh: Rectangle = Rectangle((anchor_x, anchor_y), width, height,
                                              facecolor = "none",
                                              edgecolor = "black",
                                              linestyle = "dashed")
            
            rect_thrshs += [rect_thrsh]
            
            ## Plot the spatial error
            anchor_x: float = x0 + (dx / 2.) * (1. - (col_err / err_ind.col_max_error))
            anchor_y: float = y0 + (dy / 2.) * (1. - (col_err / err_ind.col_max_error))
            width: float = dx * (col_err / err_ind.col_max_error)
            height: float = dy * (col_err / err_ind.col_max_error)
            rect_err: Rectangle = Rectangle((anchor_x, anchor_y), width, height,
                                              facecolor = "none",
                                              edgecolor = "red")
            
            rect_errs += [rect_err]

        cell_items: list = sorted(col.cells.items())
        for cell_key, cell in cell_items:
            assert(cell.is_lf)
                
            [th0, th1]   = cell.pos[:]
            [deg0, deg1] = [th0 * 180. / consts.PI, th1 * 180. / consts.PI]

            cell_err: float = err_ind.cols[col_key].cells[cell_key].error

            # Plot the element boundary
            wedge_elem: Wedge = Wedge((cx, cy), min(dx, dy)/2, deg0, deg1,
                                      facecolor = "none",
                                      edgecolor = "black")

            wedges += [wedge_elem]

            ## Plot the angular refinement threshold and angular error
            if kwargs["kind"] in ["ang", "all"]:
                ## Plot the angular refinement threshold
                radius: float = min(dx, dy) * err_ind.ang_ref_tol
                arc_thrsh: Arc = Arc((cx, cy), radius, radius, theta1 = deg0, theta2 = deg1,
                                     facecolor = "none",
                                     edgecolor = "black",
                                     linestyle = "dashed")
                
                ## Plot the angular error
                radius: float = min(dx, dy) * (cell_err / err_ind.cell_max_error)
                arc_err: Arc = Arc((cx, cy), radius, radius, theta1 = deg0, theta2 = deg1,
                                   facecolor = "none",
                                   edgecolor = "red")
                
                wedges += [arc_err, arc_thrsh]
                    
    rects: list = rect_elems + rect_thrshs + rect_errs
    rect_coll: PatchCollection = PatchCollection(rects, match_original = True)
    ax.add_collection(rect_coll)
    
    wedge_coll: PatchCollection = PatchCollection(wedges, match_original = True)
    ax.add_collection(wedge_coll)
    
    if file_path:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.tight_layout()
        plt.savefig(file_path, dpi = 300)
        plt.close(fig)

    return [fig, ax]