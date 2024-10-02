import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .plot_mesh import plot_mesh, get_prism
from .. import get_cell_nhbr_in_col

def plot_cell_nhbrs_in_col(mesh, col, cell, nhbr_col,
                           file_name = None, **kwargs):
    
    default_kwargs = {"label_cells" : False}
    kwargs = {**default_kwargs, **kwargs}

    [fig, ax] = plot_mesh(mesh, ax = None, file_name = None,
                          plot_dim = 3, **kwargs)

    if col in list(mesh.cols.values()):
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos
            if cell in list(col.cells.values()):
                [z0, z1] = cell.pos
                prism = get_prism([x0, x1], [y0, y1], [z0, z1], color = "red")
                for face in prism:
                    ax.add_collection3d(face)
                    
                    nhbr_cells = get_cell_nhbr_in_col(cell = cell,
                                                      nhbr_col = nhbr_col)

                    [x0, y0, x1, y1] = nhbr_col.pos
                    for nhbr_cell in nhbr_cells:
                        if nhbr_cell != None:
                            if nhbr_cell.is_lf:
                                [z0, z1] = nhbr_cell.pos
                            
                                prism = get_prism([x0, x1], [y0, y1], [z0, z1],
                                                  color = "blue")
                                for face in prism:
                                    ax.add_collection3d(face)
                                            
            if file_name:
                fig.set_size_inches(6.5, 6.5)
                plt.savefig(file_name, dpi = 300)
                plt.close(fig)
                
    return ax
