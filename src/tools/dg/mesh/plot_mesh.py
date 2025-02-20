# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle, Wedge

# Local Library Imports
import consts
from dg.mesh import Mesh

# Relative Imports

def plot_mesh(mesh: Mesh, file_path : str = None, **kwargs) -> list:
    default_kwargs: dict = {"lims" : [[],[]],
                            "show_p" : True, # Show ndof of each element
                            "blocking" : False # Default to non-blocking behavior for plotting
                            }

    kwargs: dict = {**default_kwargs, **kwargs}

    fig, ax = plt.subplots()

    ## Set plot range
    [Lx, Ly] = mesh.Ls[:]
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

    ## Label axes and get title
    mesh_ndof: int = mesh.get_ndof()
    title: str = "Number of Degress of Freedom: {}".format(mesh_ndof)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    ## Colors for showing ndofs for each element
    colors: list = ["#e6194B", "#f58231", "#ffe119", "#bfef45", "#aaffc3", "#3cb44b",
                    "#469990", "#42d4f4", "#4363d8", "#dcbeff", "#911eb4", "#f032e6",
                    "#fabed4", "#ffffff", "#a9a9a9", "#9A6324", "#800000"]
    unique_ndofs: list = []
    ncolors: int = len(colors)

    # Set up patches to all be added at once
    rects: list  = []
    wedges: list = []
    labels: list = []
    legend_elements: list = []

    col_items = sorted(mesh.cols.items())
    for _, col in col_items:
        assert(col.is_lf)
            
        # Determine the color of the spatial element, and add ndof to the list
        [ndof_x, _] = col.ndofs[:]
        if kwargs["show_p"]:
            color: str = colors[ndof_x%ncolors]
            if ndof_x not in unique_ndofs:
                unique_ndofs += [ndof_x]
                label: str = str(ndof_x)
                labels += [ndof_x]
                legend_elements += [Patch(facecolor = color,
                                            edgecolor = "black",
                                            label     = label)]
        else:
            color: str = "None"
        
        # Create the patch for the spatial element
        [x0, y0, x1, y1] = col.pos[:]
        [dx, dy] = [x1 - x0, y1 - y0]
        [cx, cy] = [(x0 + x1) / 2., (y0 + y1) / 2.]
        
        rect: Rectangle = Rectangle((x0, y0), dx, dy,
                                                    fill = True,
                                                    facecolor = color,
                                                    edgecolor = "black")
        rects += [rect]
        cell_items = sorted(col.cells.items())
        for _, cell in cell_items:
            assert(cell.is_lf)
                
            # Determine the color of the angular element, and add ndof to the list
            [ndof_th] = cell.ndofs[:]
            if kwargs["show_p"]:
                color: str = colors[ndof_th%ncolors]
                if ndof_th not in unique_ndofs:
                    unique_ndofs += [ndof_th]
                    label: str = str(ndof_th)
                    labels += [ndof_th]
                    legend_elements += [Patch(facecolor = color,
                                        edgecolor = "black",
                                        label     = label)]
            else:
                color: str = "None"
        
            [th0, th1] = cell.pos[:]
            [deg0, deg1] = [th0 * 180. / consts.PI, th1 * 180. / consts.PI]
            wed: Wedge = Wedge((cx, cy), min(dx, dy)/2, deg0, deg1,
                                               fill = True,
                                               facecolor = color,
                                               edgecolor = "black")
            wedges += [wed]
                    
    elem_coll: PatchCollection = PatchCollection(rects + wedges, match_original = True)
    ax.add_collection(elem_coll)
    
    if kwargs["show_p"]:
        order: np.ndarray = np.argsort(labels)
        legend_elements: np.ndarray = np.array(legend_elements)[order]
        legend_elements: list = list(legend_elements)
        ax.legend(handles = legend_elements)

    if file_path:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.tight_layout()
        plt.savefig(file_path, dpi = 300)
        plt.close(fig)

    return [fig, ax]
