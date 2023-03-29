import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .plot_mesh import plot_mesh, get_prism

def plot_col_nhbrs(mesh, col, file_name = None, **kwargs):
    
    default_kwargs = {'label_cells' : False,
                      'plot_dim' : 2}
    kwargs = {**default_kwargs, **kwargs}

    [fig, ax] = plot_mesh(mesh, ax = None, file_name = None, **kwargs)

    if kwargs['plot_dim'] == 2:
        ax = plot_col_nhbrs_2d(mesh, ax = ax, col = col,
                               file_name = file_name)
    elif kwargs['plot_dim'] == 3:
        ax = plot_col_nhbrs_3d(mesh, ax = ax, col = col,
                               file_name = file_name)
    else:
        print('Unable to plot mesh that is not 2D nor 3D')
        # TODO: Add more error handling

    return None

def plot_col_nhbrs_2d(mesh, ax = None, col = None, file_name = None):
    
    if ax:
        fig = plt.gcf()

    Lx = mesh.Ls[0]
    Ly = mesh.Ls[1]
    colors = ['#648FFF', '#DC267F', '#FE6100', '#FFB000']
    
    if col in list(mesh.cols.values()):
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos
            width = x1 - x0
            height = y1 - y0
            
            cell = Rectangle((x0, y0), width, height,
                             color = 'black', fill = True, alpha = 0.1)
            ax.add_patch(cell)
            
            for F in range(0, 4):
                [nhbr_key1, nhbr_key2] = col.nhbr_keys[F]
                for nhbr_key in [nhbr_key1, nhbr_key2]:
                    if nhbr_key != None:
                        nhbr = mesh.cols[nhbr_key]
                        if nhbr.is_lf:
                            [x0, y0, x1, y1] = nhbr.pos
                            width = x1 - x0
                            height = y1 - y0
                            
                            cell = Rectangle((x0, y0), width, height,
                                             color = colors[F],
                                             fill = True, alpha = 0.1)
                            ax.add_patch(cell)
                            
            if file_name:
                fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
                plt.savefig(file_name, dpi = 300)
                plt.close(fig)
                
    return ax

def plot_col_nhbrs_3d(mesh, ax = None, col = None,
                      file_name = None):
    if ax:
        fig = plt.gcf()
        
    if col in list(mesh.cols.values()):
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos
            prism = get_prism([x0, x1], [y0, y1], [0, 2 * np.pi], color = 'red')
            for face in prism:
                ax.add_collection3d(face)
                
            for F in range(0, 4):
                [nhbr_key1, nhbr_key2] = col.nhbr_keys[F]
                for nhbr_key in [nhbr_key1, nhbr_key2]:
                    if nhbr_key != None:
                        nhbr = mesh.cols[nhbr_key]
                        if nhbr.is_lf:
                            [x0, y0, x1, y1] = nhbr.pos
                            
                            prism = get_prism([x0, x1], [y0, y1],
                                              [0, 2 * np.pi],
                                              color = 'blue')
                            for face in prism:
                                ax.add_collection3d(face)
                            
            if file_name:
                fig.set_size_inches(6.5, 6.5)
                plt.savefig(file_name, dpi = 300)
                plt.close(fig)
                
    return ax
