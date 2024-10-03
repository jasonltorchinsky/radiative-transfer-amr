import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .plot_mesh import plot_mesh, get_prism
from .. import get_cell_nhbr_in_col

def plot_cell_nhbrs(mesh, col, cell, file_name = None, **kwargs):
    
    default_kwargs = {'label_cells' : False}
    kwargs = {**default_kwargs, **kwargs}

    [fig, ax] = plot_mesh(mesh, ax = None, file_name = None,
                          plot_dim = 3, plot_style = 'box', **kwargs)

    colors = ['#648FFF', '#DC267F', '#FE6100', '#FFB000']
    cols = list(mesh.cols.values())
    if col in cols:
        if col.is_lf:
            [x0, y0, xf, yf] = col.pos[:]
            cells = list(col.cells.values())
            if cell in cells:
                [z0, zf] = cell.pos
                prism = get_prism([x0, xf], [y0, yf], [z0, zf], color = 'black')
                for face in prism:
                    ax.add_collection3d(face)

                # Plot neighbors within same column
                for cell_nhbr_key in cell.nhbr_keys:
                    if cell_nhbr_key is not None:
                        cell_nhbr = col.cells[cell_nhbr_key]
                        
                        if cell_nhbr.is_lf:
                            [z0, zf] = cell_nhbr.pos
                            
                            prism = get_prism([x0, xf], [y0, yf], [z0, zf],
                                              color = 'blue')
                            for face in prism:
                                ax.add_collection3d(face)

                # Plot neighbors in neighboring columns
                for F in range(0, 4):
                    for nhbr_col_key in col.nhbr_keys[F]:
                        if nhbr_col_key is not None:
                            nhbr_col = mesh.cols[nhbr_col_key]
                            if nhbr_col.is_lf:
                                [x0, y0, xf, yf] = nhbr_col.pos[:]
                                nhbr_cell_keys = get_cell_nhbr_in_col(mesh,
                                                                      col.key,
                                                                      cell.key,
                                                                      nhbr_col.key)
                                for nhbr_cell_key in nhbr_cell_keys:
                                    if nhbr_cell_key is not None:
                                        nhbr_cell = nhbr_col.cells[nhbr_cell_key]
                                        if nhbr_cell.is_lf:
                                            [z0, zf] = nhbr_cell.pos
                                            
                                            prism = get_prism([x0, xf],
                                                              [y0, yf],
                                                              [z0, zf],
                                                              color = colors[F])
                                            for face in prism:
                                                ax.add_collection3d(face)
                            
            if file_name:
                fig.set_size_inches(6.5, 6.5)
                plt.savefig(file_name, dpi = 300)
                plt.close(fig)
                
    return ax
