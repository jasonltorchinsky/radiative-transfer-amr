import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


from dg.mesh import tools as mesh_tools
import dg.quadrature as qd

def plot_projection_2d(mesh, uh, file_name = None, **kwargs):
    '''
    Plot a Projection_2D object.
    '''
    
    default_kwargs = {'show_mesh': False,
                      'label_cells': False,
                      'shading': 'auto',
                      'colormap': 'Greys',
                      'title': None}
    kwargs = {**default_kwargs, **kwargs}
    
    fig, ax = plt.subplots()
    
    [Lx, Ly] = mesh.Ls[0:2]
    
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])

    cmin = 10**10
    cmax = -10**10
    for col_key, col in mesh.cols.items():
        if col.is_lf:
            proj_col = uh.cols[col_key]
            cmin = min(cmin, np.amin(proj_col.vals))
            cmax = max(cmax, np.amax(proj_col.vals))
    cmap = plt.colormaps[kwargs['colormap']]

    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos
            [dof_x, dof_y] = col.ndofs
                
            [nodes_x, _, nodes_y, _, _, _] = qd.quad_xya(dof_x, dof_y, 1)
            
            xx = x0 + (x1 - x0) / 2 * (nodes_x + 1)
            yy = y0 + (y1 - y0) / 2 * (nodes_y + 1)

            # Extract values from the column
            proj_col = uh.cols[col_key]
            proj_col_vals = np.copy(proj_col.vals)
            
            if np.size(xx) == 1:
                print('WARNING: Line 48 of plot_projection.py')
            else:
                if kwargs['shading'] == 'flat':
                    uh_col_vals = uh_col_vals[:-1,:-1]
                im = ax.pcolormesh(xx, yy, proj_col_vals,
                                   cmap = cmap,
                                   shading = kwargs['shading'],
                                   vmin = cmin, vmax = cmax)
                    
    if kwargs['show_mesh']:
        [fig, ax] = mesh_tools.plot_mesh(mesh, ax = ax, 
                                         label_cells = kwargs['label_cells'])
        
    fig.colorbar(mappable = im, ax = ax)
    
    if kwargs['title']:
        fig.suptitle(kwargs['title'])
            
    if file_name:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)
        
    return [fig, ax]

def plot_projection_3d(mesh, uh, file_name = None, **kwargs):

    default_kwargs = {'angles': [0, np.pi/2, np.pi, 3*np.pi/2],
                      'show_mesh': False,
                      'label_cells': False,
                      'shading': 'auto',
                      'colormap': 'Greys',
                      'title': None}
    kwargs = {**default_kwargs, **kwargs}

    angles = kwargs['angles']
    nangles = np.shape(angles)[0]
    [Lx, Ly] = mesh.Ls[0:2]
    
    # Set up the subplots
    [nrows, ncols] = get_closest_factors(nangles)

    fig, axs = plt.subplots(nrows, ncols, sharex = True, sharey = True)
    for ax in axs.flatten():
        ax.set_xlim([0, Lx])
        ax.set_ylim([0, Ly])

    cmin = 10**10
    cmax = -10**10
    for col_key, col in mesh.cols.items():
        if col.is_lf:
            proj_col = uh.cols[col_key]
            for cell_key, cell in col.cells.items():
                    if cell.is_lf:
                        proj_cell = proj_col.cells[cell_key]
                        cmin = min(cmin, np.amin(proj_cell.vals))
                        cmax = max(cmax, np.amax(proj_cell.vals))
    cmap = plt.colormaps[kwargs['colormap']]
    
    for a_idx in range(0, nangles):
        a = angles[a_idx]
        ax_x_idx = int(np.mod(a_idx, ncols))
        ax_y_idx = int(np.floor(a_idx / ncols))

        ax = axs[ax_y_idx, ax_x_idx]

        # Title
        a_rads = a / np.pi
        ax.set_title('{:.2f}\u03C0 Radians'.format(a_rads))
            
        for col_key, col in sorted(mesh.cols.items()):
            if col.is_lf:
                [x0, y0, x1, y1] = col.pos
                [dof_x, dof_y] = col.ndofs
                
                [nodes_x, _, nodes_y, _, _, _] = qd.quad_xya(dof_x, dof_y, 1)
                
                xx = x0 + (x1 - x0) / 2 * (nodes_x + 1)
                yy = y0 + (y1 - y0) / 2 * (nodes_y + 1)

                proj_col = uh.cols[col_key]
                
                for cell_key, cell in sorted(col.cells.items()):
                    if cell.is_lf:
                        [a0, a1] = cell.pos
                        [dof_a] = cell.ndofs
                        
                        if (a0 <= a) and (a <= a1):
                            # Extract the projected value at the desired angle using the
                            [_, _, _, _, nodes_a, _] = qd.quad_xya(1, 1, dof_a)
                            proj_cell = proj_col.cells[cell_key]
                            proj_cell_vals = np.copy(proj_cell.vals)

                            proj_cell_vals_xy = np.zeros([dof_x, dof_y])
                            for aa in range(0, dof_a):
                                proj_cell_vals_xy += proj_cell_vals[:,:,aa] \
                                    * qd.gl_eval(nodes_a, aa, a)
                                
                            if np.shape(xx)[0] == 1:
                                print('WARNING: Line 146 of plot_projection.py')
                            else:
                                if kwargs['shading'] == 'flat':
                                    proj_cell_vals_xy = proj_cell_vals_xy[:-1,:-1]
                                im = ax.pcolormesh(xx, yy, proj_cell_vals_xy,
                                                   cmap = cmap,
                                                   shading = kwargs['shading'],
                                                   vmin = cmin, vmax = cmax)
                                
        if kwargs['show_mesh']:
            [_, ax] = mesh_tools.plot_mesh(mesh, ax = ax, 
                                           label_cells = kwargs['label_cells'])
            
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
    fig.colorbar(mappable = im, cax = cbar_ax)

    if kwargs['title']:
        fig.suptitle(kwargs['title'])
    
    if file_name:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
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
    
