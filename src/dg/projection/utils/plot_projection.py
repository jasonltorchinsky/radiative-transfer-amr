import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys

sys.path.append('../..')
from dg.quadrature import quad_xyth, lag_eval
from dg.projection import push_forward, pull_back

def plot_projection(projection, file_name = None, **kwargs):
    
    default_kwargs = {'angles' : [0, np.pi/2, np.pi, 3*np.pi/2]}
    kwargs = {**default_kwargs, **kwargs}

    if not projection.has_th:
        [fig, ax] = plot_projection_2d(projection,
                                       file_name = file_name,
                                       **kwargs)
    else:
        [fig, ax] = plot_projection_3d(projection,
                                       file_name = file_name,
                                       **kwargs)
        
    return [fig, ax]

def plot_projection_2d(projection, file_name = None, **kwargs):

    default_kwargs = {}
    kwargs = {**default_kwargs, **kwargs}
    
    [Lx, Ly] = projection.Ls[:]
    
    fig, ax = plt.subplots()
    
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])

    # Get colorbar min/max
    [vmin, vmax] = [0., 0.]
    col_items = sorted(projection.cols.items())
    for col_key, col in col_items:  
        # There should only be one cell, so this should be okay
        cell_items = sorted(col.cells.items())
        for cell_key, cell in cell_items:
            vmin = min(vmin, np.amin(cell.vals))
            vmax = max(vmax, np.amax(cell.vals))
    
    for col_key, col in col_items:
        # Plot column
        [x0, y0, x1, y1] = col.pos
        [dx, dy] = [x1 - x0, y1 - y0]
        
        [ndof_x, ndof_y] = col.ndofs
        
        [xxb, _, yyb, _, _, _] = quad_xyth(nnodes_x = ndof_x,
                                           nnodes_y = ndof_y)
        
        xxf = push_forward(x0, x1, xxb)
        yyf = push_forward(y0, y1, yyb)

        ax.axvline(x = x1, color = 'black', linestyle = '-',
                   linewidth = 0.25)
        ax.axhline(y = y1, color = 'black', linestyle = '-',
                   linewidth = 0.25)
        
        # There should only be one cell, so this should be okay
        cell_items = sorted(col.cells.items())
        for cell_key, cell in cell_items:
            vals = cell.vals[:,:,0]
            
            pc = ax.pcolormesh(xxf, yyf, vals.transpose(), shading = 'auto',
                               vmin = vmin, vmax = vmax)
                
    fig.colorbar(pc)
    
    if file_name:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)

    return [fig, ax]

def plot_projection_3d(projection, file_name = None, **kwargs):

    default_kwargs = {'angles' : [0, np.pi/2, np.pi, 3*np.pi/2]}
    kwargs = {**default_kwargs, **kwargs}
    
    [Lx, Ly] = projection.Ls[:]
    Lz       = 2 * np.pi
    angles   = kwargs['angles']
    nangles  = np.shape(angles)[0]

    # Set up the subplots
    [nrows, ncols] = get_closest_factors(nangles)

    fig, axs = plt.subplots(nrows, ncols, sharex = True, sharey = True)
    for ax in axs.flatten():
        ax.set_xlim([0, Lx])
        ax.set_ylim([0, Ly])
    
    # Get colorbar min/max
    [vmin, vmax] = [10.**10, -10.**10]
    col_items = sorted(projection.cols.items())
    for col_key, col in col_items:  
        # There should only be one cell, so this should be okay
        cell_items = sorted(col.cells.items())
        for cell_key, cell in cell_items:
            vmin = min(vmin, np.amin(cell.vals))
            vmax = max(vmax, np.amax(cell.vals))

    for th_idx in range(0, nangles):
        th = angles[th_idx]
        ax_x_idx = int(np.mod(th_idx, ncols))
        ax_y_idx = int(np.floor(th_idx / ncols))

        ax = axs[ax_y_idx, ax_x_idx]

        # Title
        th_rads = th / np.pi
        ax.set_title('{:.2f}\u03C0 Radians'.format(th_rads))
        
        for col_key, col in col_items:
            # Plot column
            [x0, y0, x1, y1] = col.pos
            [dx, dy] = [x1 - x0, y1 - y0]
            
            [ndof_x, ndof_y] = col.ndofs
            
            [xxb, _, yyb, _, _, _] = quad_xyth(nnodes_x = ndof_x,
                                               nnodes_y = ndof_y)
            
            xxf = push_forward(x0, x1, xxb)
            yyf = push_forward(y0, y1, yyb)
            
            ax.axvline(x = x1, color = 'black', linestyle = '-',
                       linewidth = 0.25)
            ax.axhline(y = y1, color = 'black', linestyle = '-',
                       linewidth = 0.25)
            
            # There should only be one cell, so this should be okay
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                [th0, th1] = cell.pos
                [ndof_th]  = cell.ndofs
                if (th0 <= th) and (th <= th1):
                    [_, _, _, _, thb, _] = quad_xyth(nnodes_th = ndof_th)
                    th_pb = pull_back(th0, th1, th)
                    
                    vals_xyth = cell.vals[:,:,:]
                    vals_xy = np.zeros([ndof_x, ndof_y])
                    for ii in range(0, ndof_x):
                        for jj in range(0, ndof_y):
                            for aa in range(0, ndof_th):
                                vals_xy[ii, jj] += vals_xyth[ii, jj, aa] \
                                    * lag_eval(thb, aa, th_pb)
                    
                    pc = ax.pcolormesh(xxf, yyf, vals_xy.transpose(),
                                       shading = 'auto',
                                       vmin = vmin, vmax = vmax)

                    break
                    
    
    fig.colorbar(pc, ax = axs, location = 'right')
    
    if file_name:
        width  = (ncols / nrows) * 6.5
        height = (nrows / ncols) * 6.5
        fig.set_size_inches(width, height)
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)
        
    return [fig, ax]


def get_closest_factors(x):
    '''
    Gets the factors of x that are closest to the square root.
    '''

    a = int(np.floor(np.sqrt(x)))
    while ((x / a) - np.floor(x / a) > 10**(-3)):
        a = int(a - 1)

    return [int(a), int(x / a)]
