import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge

def plot_mesh_p(mesh, file_name = None, **kwargs):
    
    default_kwargs = {'label_cells' : False,
                      'plot_dim' : 2}
    kwargs = {**default_kwargs, **kwargs}
    
    if kwargs['plot_dim'] == 2:
        ax = plot_mesh_p_2d(mesh, file_name = file_name)
    elif kwargs['plot_dim'] == 3:
        ax = plot_mesh_p_3d(mesh, file_name = file_name)
    else:
        print('Unable to plot mesh that is not 2D nor 3D')
        # TODO: Add more error handling

    return None

def plot_mesh_p_2d(mesh, file_name = None):
    
    fig, ax = plt.subplots()

    [Lx, Ly] = mesh.Ls

    # Get colorbar min/max
    [vmin, vmax] = [10e10, -10e10]
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            [ndof_x, ndof_y] = col.ndofs[:]
            vmin = int(min(ndof_x, vmin))
            vmax = int(max(ndof_x, vmax))

    cmap = plt.get_cmap('gist_rainbow', vmax - vmin + 1)

    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos[:]
            [ndof_x, ndof_y] = col.ndofs[:]
            
            pc = ax.pcolormesh([x0, x1], [y0, y1], [[ndof_x]],
                               shading = 'flat',
                               vmin = vmin, vmax = vmax,
                               cmap = cmap, edgecolors = 'black')

    fig.colorbar(pc)
    
    if file_name:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)
                
    return ax

def plot_mesh_p_3d(mesh, file_name = None):

    fig, ax = plt.subplots()
        
    [Lx, Ly] = mesh.Ls[:]
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
    ncolors = len(colors)
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos
            [dx, dy]  = [x1 - x0, y1 - y0]
            [cx, cy] = [(x0 + x1) / 2., (y0 + y1) / 2.]

            rect = Rectangle((x0, y0), dx, dy,
                                     edgecolor = 'black',
                                     fill = None)
            ax.add_patch(rect)

            cell_items = sorted(col.cells.items())
            
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, th1] = cell.pos[:]
                    [ndof_th]  = cell.ndofs[:]

                    [deg0, deg1] = [th0 * 180. / np.pi, th1 * 180. / np.pi]
                    
                    wed = Wedge((cx, cy), min(dx, dy)/2, deg0, deg1,
                                edgecolor = 'black',
                                facecolor = colors[ndof_th%ncolors]
                                )
                    
                    ax.add_patch(wed)
            
    if file_name:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)
                
    return ax
