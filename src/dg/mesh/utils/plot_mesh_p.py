import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, Wedge
from matplotlib.collections import PatchCollection

def plot_mesh_p(mesh, file_name = None, **kwargs):
    
    default_kwargs = {'label_cells' : False,
                      'plot_dim' : 2,
                      'blocking'    : False # Defualt to non-blokcig behavior for plotting
                      }
    kwargs = {**default_kwargs, **kwargs}
    
    if kwargs['plot_dim'] == 2:
        ax = plot_mesh_p_2d(mesh, file_name = file_name, **kwargs)
    elif kwargs['plot_dim'] == 3:
        ax = plot_mesh_p_3d(mesh, file_name = file_name, **kwargs)
    else:
        print('Unable to plot mesh that is not 2D nor 3D')
        # TODO: Add more error handling

    return None

def plot_mesh_p_2d(mesh, file_name = None, **kwargs):
    
    default_kwargs = {'label_cells': False,
                      'blocking'    : False # Defualt to non-blokcig behavior for plotting
                      }
    kwargs = {**default_kwargs, **kwargs}
    
    fig, ax = plt.subplots()
        
    [Lx, Ly] = mesh.Ls[:]
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])

    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
    unique_ndof_xs = []
    ncolors = len(colors)

    rects = []
    labels = []
    legend_elements = []
    
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            [ndof_x, ndof_y] = col.ndofs[:]
            [x0, y0, x1, y1] = col.pos
            [dx, dy]  = [x1 - x0, y1 - y0]
            [cx, cy] = [(x0 + x1) / 2., (y0 + y1) / 2.]
            
            color = colors[ndof_x%ncolors]
            
            if ndof_x not in unique_ndof_xs:
                unique_ndof_xs += [ndof_x]
                label = str(ndof_x)
                labels += [ndof_x]
                legend_elements += [Patch(facecolor = color,
                                          edgecolor = 'black',
                                          label     = label)]
            
            rect = Rectangle((x0, y0), dx, dy,
                             facecolor = color,
                             edgecolor = 'black')
            rects += [rect]
            
    rect_coll = PatchCollection(rects, match_original = True)
    ax.add_collection(rect_coll)
    
    order = np.argsort(labels)
    legend_elements = np.array(legend_elements)[order]
    legend_elements = list(legend_elements)
    ax.legend(handles = legend_elements)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    if file_name:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.tight_layout()
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)
                
    return ax

def plot_mesh_p_3d(mesh, file_name = None, **kwargs):
                      
    default_kwargs = {'label_cells': False,
                      'blocking'    : False # Defualt to non-blokcig behavior for plotting
                      }
    kwargs = {**default_kwargs, **kwargs}
                      
    fig, ax = plt.subplots()
        
    [Lx, Ly] = mesh.Ls[:]
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
    unique_ndof_ths = []
    ncolors = len(colors)

    wedges = []
    rects = []
    labels = []
    legend_elements = []
    
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos
            [dx, dy]  = [x1 - x0, y1 - y0]
            [cx, cy] = [(x0 + x1) / 2., (y0 + y1) / 2.]

            rect = Rectangle((x0, y0), dx, dy,
                             edgecolor = 'black',
                             fill = None)
            rects += [rect]

            cell_items = sorted(col.cells.items())
            
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, th1] = cell.pos[:]
                    [ndof_th]  = cell.ndofs[:]

                    [deg0, deg1] = [th0 * 180. / np.pi, th1 * 180. / np.pi]
                    color = colors[ndof_th%ncolors]
                    
                    if ndof_th not in unique_ndof_ths:
                        unique_ndof_ths += [ndof_th]
                        label = str(ndof_th)
                        labels += [ndof_th]
                        legend_elements += [Patch(facecolor = color,
                                                  edgecolor = 'black',
                                                  label     = label)]
                    
                    wedge = Wedge((cx, cy), min(dx, dy)/2, deg0, deg1,
                                  edgecolor = 'black',
                                  facecolor = color
                                  )
                    
                    wedges += [wedge]
                    
    rect_coll = PatchCollection(rects, match_original = True)
    ax.add_collection(rect_coll)
    
    wedge_coll = PatchCollection(wedges, match_original = True)
    ax.add_collection(wedge_coll)
    
    order = np.argsort(labels)
    legend_elements = np.array(legend_elements)[order]
    legend_elements = list(legend_elements)
    ax.legend(handles = legend_elements)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    if file_name:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.tight_layout()
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)
                
    return ax
