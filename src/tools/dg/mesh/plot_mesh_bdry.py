import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .plot_mesh import plot_mesh, get_prism

def plot_mesh_bdry(mesh, file_name = None, **kwargs):
    
    default_kwargs = {'label_cells' : False,
                      'plot_dim' : 2}
    kwargs = {**default_kwargs, **kwargs}

    [fig, ax] = plot_mesh(mesh, ax = None, file_name = None, **kwargs)
    
    if kwargs['plot_dim'] == 2:
        ax = plot_mesh_bdry_2d(mesh, ax = ax,
                               file_name = file_name)
    elif kwargs['plot_dim'] == 3:
        ax = plot_mesh_bdry_3d(mesh, ax = ax,
                               file_name = file_name)
    else:
        print('Unable to plot mesh that is not 2D nor 3D')
        # TODO: Add more error handling

    return None

def plot_mesh_bdry_2d(mesh, ax = None, file_name = None):
    
    if ax:
        fig = plt.gcf()
        
    [Lx, Ly] = mesh.Ls[:]
    
    colors = ['#648FFF', '#DC267F', '#FE6100', '#FFB000']
    for col in list(mesh.cols.values()):
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos
            [dx, dy]  = [x1 - x0, y1 - y0]
            
            # Choose color dependent on which boundary the column is on
            is_bdry = False
            for F in range(0, 4): # [Right, Top, Left, Bottom]
                if ((col.nhbr_keys[F][0] is None) and
                    (col.nhbr_keys[F][1] is None)):
                    color = colors[F]
                    rect = Rectangle((x0, y0), dx, dy,
                                     edgecolor = 'black',
                                     facecolor = color,
                                     alpha = 0.2)
                    ax.add_patch(rect)

                    is_bdry = True
            
            if not is_bdry: # In the interior
                rect = Rectangle((x0, y0), dx, dy,
                                 edgecolor = 'black',
                                 fill = None)
                ax.add_patch(rect)
            
    if file_name:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)
                
    return ax

def plot_mesh_bdry_3d(mesh, ax = None,
                      file_name = None):
    if ax:
        fig = plt.gcf()
        
    [Lx, Ly] = mesh.Ls[:]
        
    colors = ['#648FFF', '#DC267F', '#FE6100', '#FFB000']
    inflow_bdrys = [[1, 2], [2, 3], [3, 0], [0, 1]]
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos
            [dx, dy]  = [x1 - x0, y1 - y0]
            [cx, cy] = [(x0 + x1) / 2., (y0 + y1) / 2.]

            cell_items = sorted(col.cells.items())
            
            rect = Rectangle((x0, y0), dx, dy,
                                     edgecolor = 'black',
                                     fill = None)
            ax.add_patch(rect)
            
            col_is_bdry = False
            for F in range(0, 4): # [Right, Top, Left, Bottom]
                if ((col.nhbr_keys[F][0] is None) and
                    (col.nhbr_keys[F][1] is None)):
                    # Choose color dependent on which boundary the column is on
                    color = colors[F]
                    
                    col_is_bdry = True
                    
                    inflow_bdry = inflow_bdrys[F]
                    
                    for cell_key, cell in cell_items:
                        if cell.is_lf:
                            [th0, th1] = cell.pos[:]
                            [deg0, deg1] = [th0 * 180. / np.pi, th1 * 180. / np.pi]
                            quad = cell.quad

                            if quad in inflow_bdry:
                                wed = Wedge((cx, cy), min(dx, dy)/2, deg0, deg1,
                                            edgecolor = 'black',
                                            facecolor = color,
                                            alpha = 0.2
                                            )
                            else:
                                wed = Wedge((cx, cy), min(dx, dy)/2, deg0, deg1,
                                            edgecolor = 'black',
                                            fill = None
                                            )
                            ax.add_patch(wed)
            
            if not col_is_bdry: # In the interior
                for cell_key, cell in cell_items:
                        if cell.is_lf:
                            [th0, th1] = cell.pos[:]
                            [deg0, deg1] = [th0 * 180. / np.pi, th1 * 180. / np.pi]
                            
                            wed = Wedge((cx, cy), min(dx, dy)/2, deg0, deg1,
                                        edgecolor = 'black',
                                        fill = None
                                        )
                            ax.add_patch(wed)
            
    if file_name:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)
                
    return ax
