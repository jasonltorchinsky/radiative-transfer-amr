import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys

sys.path.append('../..')
from dg.quadrature import quad_xyth
from dg.projection import push_forward

def plot_projection(projection, ax = None, file_name = None, **kwargs):
    
    default_kwargs = {'label_cells' : False,
                      'plot_dim' : 2}
    kwargs = {**default_kwargs, **kwargs}

    if kwargs['plot_dim'] == 2:
        [fig, ax] = plot_projection_2d(projection, ax = ax,
                                       file_name = file_name, **kwargs)
    elif kwargs['plot_dim'] == 3:
        [fig, ax] = plot_projection_3d(projection, ax = ax,
                                       file_name = file_name, **kwargs)
    else:
        print('Unable to plot mesh that is not 2D nor 3D')
        # TODO: Add more error handling

    return [fig, ax]

def plot_projection_2d(projection, ax = None, file_name = None, **kwargs):

    default_kwargs = {'label_cells': False}
    kwargs = {**default_kwargs, **kwargs}
    
    [Lx, Ly] = projection.Ls[:]

    if ax:
        fig = plt.gcf()
    else:
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

    col_items = sorted(projection.cols.items())
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
            
            # Label each cell with idxs, lvs
            if kwargs['label_cells']:
                idx = col.idx
                lv  = col.lv
                x_mid = (x0 + x1) / 2.
                y_mid = (y0 + y1) / 2.
                label = '{}, {}'.format(idx, lv)
                label = '{}'.format(col.key)
                ax.text(x_mid, y_mid, label,
                        ha = 'center', va = 'center')
                
    fig.colorbar(pc)
    
    if file_name:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)

    return [fig, ax]

def plot_projection_3d(projection, ax = None, file_name = None, **kwargs):

    default_kwargs = {'label_cells': False}
    kwargs = {**default_kwargs, **kwargs}
    
    Lx = mesh.Ls[0]
    Ly = mesh.Ls[1]
    Lz = 2 * np.pi

    if ax:
        fig = plt.gcf()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection = '3d')
    
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    ax.set_zlim([0, Lz])
    
    for col in list(mesh.cols.values()):
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos
            for cell in list(col.cells.values()):
                # Plot cell
                [z0, z1] = cell.pos
                prism = get_prism([x0, x1], [y0, y1], [z0, z1], color = 'w')
                for face in prism:
                    ax.add_collection3d(face)
                    
                # Label each cell with idxs, lvs
                if kwargs['label_cells']:
                    idx = col.idx
                    lv  = col.lv
                    xmid = (x0 + x1) / 2.
                    ymid = (y0 + y1) / 2.
                    zmid = (z0 + z1) / 2.
                    label = '{}, {}'.format(idx, lv)
                    ax.text(xmid, ymid, zmid, label, zdir = None)
        
    if file_name:
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)

    return [fig, ax]

def get_xy_face(xs, ys, z):
    verts = 4 * [0]
    verts[0] = [xs[0], ys[0], z]
    verts[1] = [xs[0], ys[1], z]
    verts[2] = [xs[1], ys[1], z]
    verts[3] = [xs[1], ys[0], z]
    
    return [verts]

def get_yz_face(x, ys, zs):
    verts = 4 * [0]
    verts[0] = [x, ys[0], zs[0]]
    verts[1] = [x, ys[0], zs[1]]
    verts[2] = [x, ys[1], zs[1]]
    verts[3] = [x, ys[1], zs[0]]
    
    return [verts]

def get_xz_face(xs, y, zs):
    verts = 4 * [0]
    verts[0] = [xs[0], y, zs[0]]
    verts[1] = [xs[0], y, zs[1]]
    verts[2] = [xs[1], y, zs[1]]
    verts[3] = [xs[1], y, zs[0]]
    
    return [verts]

def get_face(xs, ys, zs):
    if type(xs) != list:
        verts = get_yz_face(xs, ys, zs)
    elif type(ys) != list:
        verts = get_xz_face(xs, ys, zs)
    elif type(zs) != list:
        verts = get_xy_face(xs, ys, zs)
        
    return verts

def get_prism(xs, ys, zs, color = 'w'):
    faces_verts = 6 * [0]
    for ii in range(0, 2):
        faces_verts[ii] = get_face(xs[ii], ys, zs)

    for jj in range(0, 2):
        faces_verts[2 + jj] = get_face(xs, ys[jj], zs)

    for kk in range(0, 2):
        faces_verts[4 + kk] = get_face(xs, ys, zs[kk])

    faces = 6 * [0]
    for idx in range(0, 6):
        verts = faces_verts[idx]
        faces[idx] = Poly3DCollection(verts,
                                      facecolors = color,
                                      edgecolor = 'k',
                                      alpha = 0.1)
        
    return faces

def plot_mesh_upg(mesh, ax = None, file_name = None, **kwargs):
    default_kwargs = {'label_cells': False,
                      'show_attr': None,
                      'colormap': 'viridis'}
    kwargs = {**default_kwargs, **kwargs}

    if type(kwargs['show_attr']) is not list:
        [fig, ax] = plot_mesh_attr(mesh, ax = ax,
                                   file_name = file_name,
                                   **kwargs)
    else:
        if ax is not None:
            print(( 'ERROR IN PLOTTING MESH: Unable to add mesh' +
                    ' plot of multiple attributes ot existing axis.' +
                    ' Aborting...' ))
            sys.exit(13)
        if mesh.ndim != 2:
            print(( 'ERROR IN PLOTTING MESH: Unable to plot mesh' +
                    ' with more or fewer than two dimensions.' +
                    ' Aborting...' ))
            sys.exit(13)
        
        Lx = mesh.Ls[0]
        Ly = mesh.Ls[1]
        
        # Set up the subplots
        nattrs = len(kwargs['show_attr'])
        [nrows, ncols] = get_closest_factors(nattrs)
        
        fig, axs = plt.subplots(nrows, ncols, sharex = True, sharey = True)
        for ax in axs.flatten():
            ax.set_xlim([0, Lx])
            ax.set_ylim([0, Ly])

        attrs = kwargs['show_attr']
        for ax_idx in range(0, nattrs):
            attr = attrs[ax_idx]
            kwargs['show_attr'] = attr
            
            ax_x_idx = int(np.mod(ax_idx, ncols))
            ax_y_idx = int(np.floor(ax_idx / ncols))
            ax = axs[ax_y_idx, ax_x_idx]
            
            plot_mesh_attr(mesh, ax = ax, file_name = None, **kwargs)
            
        if file_name:
            fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
            plt.savefig(file_name, dpi = 300)
            plt.close(fig)
            

    return [fig, ax]

def plot_mesh_attr(mesh, ax = None, file_name = None, **kwargs):

    default_kwargs = {'label_cells': False,
                      'show_attr': None,
                      'colormap': 'viridis'}
    kwargs = {**default_kwargs, **kwargs}
    
    [Lx, Ly] = mesh.Ls[0:2]

    if ax:
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots()

    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])

    nrects = 0
    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            nrects += 1

    # If we are plotting an attribute, get the color scale necessary
    if kwargs['show_attr']:
        cmap = plt.colormaps[kwargs['colormap']]
        cmin = 10**10
        cmax = -10**10
        if kwargs['show_attr'] == 'dof_x':
            for col_key, col in sorted(mesh.cols.items()):
                if col.is_lf:
                    cmin = np.amin([cmin, col.ndofs[0]])
                    cmax = np.amax([cmax, col.ndofs[0]])
        elif kwargs['show_attr'] == 'dof_y':
            for col_key, col in sorted(mesh.cols.items()):
                if col.is_lf:
                    cmin = np.amin([cmin, col.ndofs[0]])
                    cmax = np.amax([cmax, col.ndofs[0]])
        #elif kwargs['show_attr'] == 'dof_a':
        #    for col_key, col in sorted(mesh.cols.items()):
        #        if col.is_lf:
        #            for cell_key, cell in sorted(col.cells.items()):
        #                cmin = np.amin(cmin, cell.ndofs[0])
        #                cmax = np.amax(cmax, cell.ndofs[0])
        elif kwargs['show_attr'] == 'lv':
            for col_key, col in sorted(mesh.cols.items()):
                if col.is_lf:
                    cmin = np.amin([cmin, col.lv])
                    cmax = np.amax([cmax, col.lv])

        cmin -= 0.1
        cmax += 0.1
    
    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            # Plot cell
            [x0, y0, x1, y1] = col.pos
            width = x1 - x0
            height = y1 - y0
            
            rect = Rectangle((x0, y0), width, height, fill = False)
            ax.add_patch(rect)
            
            if kwargs['show_attr']:
                xx = np.asarray([x0, x1])
                yy = np.asarray([y0, y1])
                if kwargs['show_attr'] == 'dof_x':
                    zz = np.asarray([[col.ndofs[0]]])
                elif kwargs['show_attr'] == 'dof_y':
                    zz = np.asarray([[col.ndofs[1]]])
                #elif kwargs['show_attr'] == 'dof_a':
                #    zz = np.asarray([[mesh.dof_a[key]]])
                elif kwargs['show_attr'] == 'lv':
                    zz = np.asarray([[col.lv]])
                im = ax.pcolormesh(xx, yy, zz,
                                   cmap = cmap,
                                   shading = 'flat',
                                   vmin = cmin, vmax = cmax)
                
            # Label each cell with idxs, lvs
            if kwargs['label_cells']:
                idxs = col.idx
                lvs  = col.lv
                xmid = (x0 + x1) / 2.
                ymid = (y0 + y1) / 2.
                label = str(idx) + ', ' + str(lv)
                ax.text(xmid, ymid, label,
                        ha = 'center', va = 'center')

    if kwargs['show_attr']:
        if file_name:
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
            cbar = fig.colorbar(mappable = im, cax = cbar_ax)
        else:
            cbar = fig.colorbar(mappable = im, ax = ax)
            
        cbar.set_ticks(np.arange(np.ceil(cmin), np.floor(cmax) + 1))
        if kwargs['show_attr'] == 'dof_x':
            cbar.set_label('Degrees of Freedom [x]',
                           rotation = 270,
                           labelpad = 15)
        elif kwargs['show_attr'] == 'dof_y':
            cbar.set_label('Degrees of Freedom [y]',
                           rotation = 270,
                           labelpad = 15)
        #elif kwargs['show_attr'] == 'dof_a':
        #    cbar.set_label('Degrees of Freedom [a]',
        #                   rotation = 270,
        #                   labelpad = 15)
        elif kwargs['show_attr'] == 'lv':
            cbar.set_label('Refinement Level [lv]',
                           rotation = 270,
                           labelpad = 15)
        
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
