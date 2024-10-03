# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import matplotlib.collections as collections
import matplotlib.pyplot  as plt
import matplotlib.patches as patches

# Local Library Imports
from dg.mesh import Mesh

# Relative Imports

def plot_mesh(mesh: Mesh, lims: list = [[],[]], file_path : str = None,
                  **kwargs):
    default_kwargs: dict = {"label_cells" : False,
                            "plot_dim"    : 2,
                            "plot_style"  : "flat",
                            "blocking"    : False # Defualt to non-blokcig behavior for plotting
                            }

    kwargs: dict = {**default_kwargs, **kwargs}

    # Set up figure, and plotting range
    fig, ax = plt.subplots()

    [Lx, Ly] = mesh.Ls[:]
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

    # Set up colors for showing the ndof of each element
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
        if col.is_lf:
            # Determine the color of the spatial element, and add ndof to the list
            [ndof_x, _] = col.ndofs[:]
            color: str = colors[ndof_x%ncolors]
            if ndof_x not in unique_ndofs:
                unique_ndofs += [ndof_x]
                label: str = str(ndof_x)
                labels += [ndof_x]
                legend_elements += [patches.Patch(facecolor = color,
                                            edgecolor = "black",
                                            label     = label)]
            
            # Create the patch for the spatial element
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy] = [x1 - x0, y1 - y0]
            [cx, cy] = [(x0 + x1) / 2., (y0 + y1) / 2.]
            
            rect: patches.Rectangle = patches.Rectangle((x0, y0), dx, dy,
                                                        fill = True,
                                                        facecolor = color,
                                                        edgecolor = "black")
            rects += [rect]

            cell_items = sorted(col.cells.items())
            for _, cell in cell_items:
                if cell.is_lf:
                    # Determine the color of the spatial element, and add ndof to the list
                    [ndof_th] = cell.ndofs[:]
                    color: str = colors[ndof_th%ncolors]
                    if ndof_th not in unique_ndofs:
                        unique_ndofs += [ndof_th]
                        label: str = str(ndof_th)
                        labels += [ndof_th]
                        legend_elements += [patches.Patch(facecolor = color,
                                            edgecolor = "black",
                                            label     = label)]
                
                    [th0, th1] = cell.pos[:]
                    [deg0, deg1] = [th0 * 180. / np.pi, th1 * 180. / np.pi]

                    wed: patches.Wedge = patches.Wedge((cx, cy), min(dx, dy)/2, deg0, deg1,
                                                       fill = True,
                                                       facecolor = color,
                                                       edgecolor = "black")
                    wedges += [wed]
                    
    elem_coll: collections.PatchCollection = collections.PatchCollection(rects + wedges, match_original = True)
    ax.add_collection(elem_coll)
    
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
