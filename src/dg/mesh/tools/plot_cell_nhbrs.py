import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .plot_mesh import plot_mesh, get_prism

def plot_cell_nhbrs(mesh, col, cell, file_name = None, **kwargs):
    
    default_kwargs = {'label_cells' : False}
    kwargs = {**default_kwargs, **kwargs}

    [fig, ax] = plot_mesh(mesh, ax = None, file_name = None,
                          plot_dim = 3, **kwargs)

    cols = list(mesh.cols.values())
    if col in cols:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos
            cells = list(col.cells.values())
            if cell in cells:
                [z0, z1] = cell.pos
                prism = get_prism([x0, x1], [y0, y1], [z0, z1], color = 'red')
                for face in prism:
                    ax.add_collection3d(face)
                    
                for cell_nhbr_key in cell.nhbr_keys:
                    if cell_nhbr_key is not None:
                        cell_nhbr = col.cells[cell_nhbr_key]
                        
                        if cell_nhbr.is_lf:
                            [z0, z1] = cell_nhbr.pos
                            
                            prism = get_prism([x0, x1], [y0, y1], [z0, z1],
                                              color = 'blue')
                            for face in prism:
                                ax.add_collection3d(face)
                            
            if file_name:
                fig.set_size_inches(6.5, 6.5)
                plt.savefig(file_name, dpi = 300)
                plt.close(fig)
                
    return ax
