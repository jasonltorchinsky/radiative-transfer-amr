import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

sys.path.append('../..')
from dg.quadrature import quad_xyth, lag_eval
from dg.projection import push_forward, pull_back

def plot_projection(mesh, proj, file_name = None, **kwargs):
    
    default_kwargs = {'angles' : [0, np.pi/2, np.pi, 3*np.pi/2],
                      'cmap' : 'hot'}
    kwargs = {**default_kwargs, **kwargs}

    if not mesh.has_th:
        [fig, ax] = plot_projection_2d(mesh, proj,
                                       file_name = file_name,
                                       **kwargs)
    else:
        [fig, ax] = plot_projection_3d(mesh, proj,
                                       file_name = file_name,
                                       **kwargs)
        
    return [fig, ax]

def plot_projection_2d(mesh, proj, file_name = None, **kwargs):

    default_kwargs = {}
    kwargs = {**default_kwargs, **kwargs}
    
    [Lx, Ly] = mesh.Ls[:]
    
    fig, ax = plt.subplots()
    
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])

    # Get colorbar min/max
    [vmin, vmax] = [0., 0.]
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    proj_cell = proj.cols[col_key].cells[cell_key]
                    vmin = min(vmin, np.amin(proj_cell.vals))
                    vmax = max(vmax, np.amax(proj_cell.vals))
    
    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos[:]
            [ndof_x, ndof_y] = col.ndofs[:]
            
            [xxb, _, yyb, _, _, _] = quad_xyth(nnodes_x = ndof_x,
                                               nnodes_y = ndof_y)
            
            xxf = push_forward(x0, x1, xxb)
            yyf = push_forward(y0, y1, yyb)
        
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    proj_cell = proj.cols[col_key].cells[cell_key]
                    vals = proj_cell.vals[:, :, 0]
                    
                    pc = ax.pcolormesh(xxf, yyf, vals.transpose(),
                                       shading = 'gouraud',
                                       vmin = vmin, vmax = vmax,
                                       cmap = kwargs['cmap'])
                    
                    
    for col_key, col in col_items:
        if col.is_lf:
            # Plot column boundary
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy]         = [x1 - x0, y1 - y0]
            
            rect = Rectangle((x0, y0), dx, dy, fill = False)
            ax.add_patch(rect)
            
    fig.colorbar(pc)
    
    if file_name:
        fig.set_size_inches(12, 12 * (Ly / Lx))
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)

    return [fig, ax]

def plot_projection_3d(mesh, proj, file_name = None, **kwargs):

    default_kwargs = {'angles' : [0, np.pi/2, np.pi, 3*np.pi/2]}
    kwargs = {**default_kwargs, **kwargs}
    
    [Lx, Ly] = mesh.Ls[:]
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
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    proj_cell = proj.cols[col_key].cells[cell_key]
                    vmin = min(vmin, np.amin(proj_cell.vals))
                    vmax = max(vmax, np.amax(proj_cell.vals))

    for th_idx in range(0, nangles):
        th = angles[th_idx]
        ax_x_idx = int(np.mod(th_idx, ncols))
        ax_y_idx = int(np.floor(th_idx / ncols))

        ax = axs[ax_y_idx, ax_x_idx]

        # Title
        th_rads = th / np.pi
        ax.set_title('{:.2f}\u03C0 Radians'.format(th_rads))
        
        for col_key, col in col_items:
            if col.is_lf:
                [x0, y0, x1, y1] = col.pos[:]
                [ndof_x, ndof_y] = col.ndofs[:]
                
                [xxb, _, yyb, _, _, _] = quad_xyth(nnodes_x = ndof_x,
                                                   nnodes_y = ndof_y)
                
                xxf = push_forward(x0, x1, xxb)
                yyf = push_forward(y0, y1, yyb)
                
                cell_items = sorted(col.cells.items())
                for cell_key, cell in cell_items:
                    if cell.is_lf:
                        [th0, th1] = cell.pos[:]
                        if (th0 <= th) and (th <= th1):
                            [ndof_th]  = cell.ndofs[:]
                            [_, _, _, _, thb, _] = quad_xyth(nnodes_th = ndof_th)
                            th_pb = pull_back(th0, th1, th)

                            proj_cell = proj.cols[col_key].cells[cell_key]
                            
                            vals_xyth = proj_cell.vals[:,:,:]
                            vals_xy = np.zeros([ndof_x, ndof_y])
                            for ii in range(0, ndof_x):
                                for jj in range(0, ndof_y):
                                    for aa in range(0, ndof_th):
                                        vals_xy[ii, jj] += vals_xyth[ii, jj, aa] \
                                            * lag_eval(thb, aa, th_pb)
                                        
                            pc = ax.pcolormesh(xxf, yyf, vals_xy.transpose(),
                                               shading = 'gouraud',
                                               vmin = vmin, vmax = vmax,
                                               cmap = kwargs['cmap'])
                            
                            break
    
    for th_idx in range(0, nangles):
        th = angles[th_idx]
        ax_x_idx = int(np.mod(th_idx, ncols))
        ax_y_idx = int(np.floor(th_idx / ncols))
        
        ax = axs[ax_y_idx, ax_x_idx]
        
        # Title
        th_rads = th / np.pi
        ax.set_title('{:.2f}\u03C0 Radians'.format(th_rads))
        
        for col_key, col in col_items:
            if col.is_lf:
                # Plot column
                [x0, y0, x1, y1] = col.pos[:]
                [dx, dy]         = [x1 - x0, y1 - y0]
                
                rect = Rectangle((x0, y0), dx, dy, fill = False)
                ax.add_patch(rect)
            
    fig.colorbar(pc, ax = axs, location = 'right')
    
    if file_name:
        width  = (ncols / nrows) * 12
        height = (nrows / ncols) * 12
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
