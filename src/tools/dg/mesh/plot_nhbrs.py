# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Wedge

# Local Library Imports
import consts
from dg.mesh import Mesh
from dg.mesh.column import Column
from dg.mesh.column.cell import Cell

# Relative Imports

def plot_nhbrs(mesh: Mesh, col_key: int, cell_key: int,
               file_path : str = None, **kwargs) -> list:
    default_kwargs: dict = {"lims" : [[],[]],
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

    ## Get all element neighbors

    ## Start with column neighbors because we will certainly want them
    col: Column = mesh.cols[col_key]
    assert(col.is_lf)

    nhbr_col_keys: list = list(set([key for x in col.nhbr_keys for key in x if key is not None]))

    if cell_key is not None: ## If we want neighbors of a cell
        cell: Cell = col.cells[cell_key]
        assert(cell.is_lf)

        intra_nhbr_keys: list = [[col_key, cell.nhbr_keys[0]],
                                 [col_key, cell.nhbr_keys[1]]]
        inter_nhbr_keys: list = []

        for nhbr_col_key in nhbr_col_keys:
            nhbr_cell_keys: list = mesh.nhbr_cells_in_nhbr_col(col_key, cell_key, nhbr_col_key)
            for nhbr_cell_key in nhbr_cell_keys:
                if nhbr_cell_key is not None:
                    inter_nhbr_keys += [[nhbr_col_key, nhbr_cell_key]]

    ## Colors for showing self, neighbors in same col (for cell)
    ## and neighbor in different col
    self_color: str = "#E69F00"
    intra_nhbr_color: str = "#56B4E9"
    inter_nhbr_color: str = "#009E73"

    # Set up patches to all be added at once
    rects: list  = []
    wedges: list = []

    col_items = sorted(mesh.cols.items())
    for col_key_i, col_i in col_items:
        assert(col_i.is_lf)
        
        # Create the patch for the spatial element
        [x0, y0, x1, y1] = col_i.pos[:]
        [dx, dy] = [x1 - x0, y1 - y0]
        [cx, cy] = [(x0 + x1) / 2., (y0 + y1) / 2.]
        
        if cell_key is None:
            if col_key_i == col_key:
                color: str = self_color
            elif col_key_i in nhbr_col_keys:
                color: str = inter_nhbr_color
            else:
                color: str = "none"
        else:
            color: str = "none"

        rect: Rectangle = Rectangle((x0, y0), dx, dy,
                                    fill = True,
                                    facecolor = color,
                                    edgecolor = "black")
        rects += [rect]
        cell_items = sorted(col_i.cells.items())
        for cell_key_i, cell_i in cell_items:
            assert(cell_i.is_lf)
        
            if cell_key is not None:
                if [col_key_i, cell_key_i] == [col_key, cell_key]:
                    color: str = self_color
                elif [col_key_i, cell_key_i] in intra_nhbr_keys:
                    color: str = intra_nhbr_color
                elif [col_key_i, cell_key_i] in inter_nhbr_keys:
                    color: str = inter_nhbr_color
                else:
                    color: str = "none"
            else:
                color: str = "none"

            [th0, th1] = cell_i.pos[:]
            [deg0, deg1] = [th0 * 180. / consts.PI, th1 * 180. / consts.PI]
            wed: Wedge = Wedge((cx, cy), min(dx, dy)/2, deg0, deg1,
                                fill = True,
                                facecolor = color,
                                edgecolor = "black")
            wedges += [wed]
                    
    elem_coll: PatchCollection = PatchCollection(rects + wedges, match_original = True)
    ax.add_collection(elem_coll)

    if file_path:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.tight_layout()
        plt.savefig(file_path, dpi = 300)
        plt.close(fig)

    return [fig, ax]
