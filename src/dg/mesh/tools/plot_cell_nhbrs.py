import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .plot_mesh import plot_mesh, get_prism
from ..ji_mesh import get_cell_nhbr

def plot_cell_nhbrs(mesh, col, cell, file_name = None, **kwargs):
    
    default_kwargs = {'label_cells' : False}
    kwargs = {**default_kwargs, **kwargs}

    [fig, ax] = plot_mesh(mesh, ax = None, file_name = None,
                          plot_dim = 3, **kwargs)

    if col in list(mesh.cols.values()):
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos
            if cell in list(col.cells.values()):
                [z0, z1] = cell.pos
                prism = get_prism([x0, x1], [y0, y1], [z0, z1], color = 'red')
                for face in prism:
                    ax.add_collection3d(face)
                    
                nhbr_locs = ['+', '-']
                for nhbr_loc in nhbr_locs:
                    nhbr = get_cell_nhbr(col = col, cell = cell,
                                         nhbr_loc = nhbr_loc)
                    if nhbr != None:
                        [z0, z1] = nhbr.pos
                        
                        prism = get_prism([x0, x1], [y0, y1], [z0, z1],
                                          color = 'blue')
                        for face in prism:
                            ax.add_collection3d(face)
                            
            if file_name:
                fig.set_size_inches(6.5, 6.5)
                plt.savefig(file_name, dpi = 300)
                plt.close(fig)
                
    return ax
