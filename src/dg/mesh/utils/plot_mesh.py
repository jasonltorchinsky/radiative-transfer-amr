import numpy                      as np
import matplotlib.pyplot          as plt
import sys
from   matplotlib.patches         import Rectangle, Wedge
from   mpl_toolkits.mplot3d       import Axes3D
from   mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_mesh(mesh, ax = None, file_name = None, **kwargs):
    
    default_kwargs = {"label_cells" : False,
                      "plot_dim"    : 2,
                      "plot_style"  : "flat",
                      "blocking"    : False # Defualt to non-blokcig behavior for plotting
                      }
    kwargs = {**default_kwargs, **kwargs}

    if kwargs["plot_dim"] == 2:
        [fig, ax] = plot_mesh_2d(mesh, ax = ax, file_name = file_name, **kwargs)
    elif kwargs["plot_dim"] == 3 and kwargs["plot_style"] == "box":
        [fig, ax] = plot_mesh_3d_box(mesh, ax = ax,
                                     file_name = file_name,
                                     **kwargs)
    elif kwargs["plot_dim"] == 3 and kwargs["plot_style"] == "flat":
        [fig, ax] = plot_mesh_3d_flat(mesh, ax = ax,
                                      file_name = file_name,
                                      **kwargs)
    else:
        print("Unable to plot mesh that is not 2D nor 3D")
        # TODO: Add more error handling

    return [fig, ax]

def plot_mesh_2d(mesh, ax = None, file_name = None, **kwargs):

    default_kwargs = {"label_cells": False,
                      "blocking"    : False # Defualt to non-blokcig behavior for plotting
                      }
    kwargs = {**default_kwargs, **kwargs}
    
    Lx = mesh.Ls[0]
    Ly = mesh.Ls[1]

    if ax:
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots()

    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])

    for col in list(mesh.cols.values()):
        if col.is_lf:
            # Plot cell
            [x0, y0, x1, y1] = col.pos
            width = x1 - x0
            height = y1 - y0
            
            cell = Rectangle((x0, y0), width, height, fill = False)
            ax.add_patch(cell)
            
            # Label each cell with idxs, lvs
            if kwargs["label_cells"]:
                idx = col.idx
                lv  = col.lv
                x_mid = (x0 + x1) / 2.
                y_mid = (y0 + y1) / 2.
                label = "{}, {}".format(idx, lv)
                label = "{}".format(col.key)
                ax.text(x_mid, y_mid, label,
                        ha = "center", va = "center")
                
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    if file_name:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.tight_layout()
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)
        
    return [fig, ax]

def plot_mesh_3d_box(mesh, ax = None, file_name = None, **kwargs):

    default_kwargs = {"label_cells": False,
                      "blocking"    : False # Defualt to non-blokcig behavior for plotting
                      }
    kwargs = {**default_kwargs, **kwargs}
    
    Lx = mesh.Ls[0]
    Ly = mesh.Ls[1]
    Lz = 2 * np.pi

    if ax:
        fig = plt.gcf()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection = "3d")
    
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    ax.set_zlim([0, Lz])
    
    for col in list(mesh.cols.values()):
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos
            for cell in list(col.cells.values()):
                # Plot cell
                [z0, z1] = cell.pos
                prism = get_prism([x0, x1], [y0, y1], [z0, z1], color = "w")
                for face in prism:
                    ax.add_collection3d(face)
                    
                # Label each cell with idxs, lvs
                if kwargs["label_cells"]:
                    idx = col.idx
                    lv  = col.lv
                    xmid = (x0 + x1) / 2.
                    ymid = (y0 + y1) / 2.
                    zmid = (z0 + z1) / 2.
                    label = "{}, {}".format(idx, lv)
                    ax.text(xmid, ymid, zmid, label, zdir = None)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(r"$\theta$")
    
    nth_ticks = 9
    th_ticks = np.linspace(0, 2, nth_ticks) * np.pi
    th_tick_labels = [None] * nth_ticks
    for aa in range(0, nth_ticks):
        th_rad = th_ticks[aa] / np.pi
        th_tick_labels[aa] = "{:.2f}\u03C0".format(th_rad)
    ax.set_zticks(th_ticks)
    ax.set_zticklabels(th_tick_labels)
    
    if file_name:
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(file_name, dpi = 300, bbox_inches = "tight")
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

def get_prism(xs, ys, zs, color = "w"):
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
                                      edgecolor = "k",
                                      alpha = 0.1)
        
    return faces

def get_closest_factors(x):
    """
    Gets the factors of x that are closest to the square root.
    """

    a = int(np.floor(np.sqrt(x)))
    while ((x / a) - np.floor(x / a) > 10**(-3)):
        a = int(a - 1)

    return [int(a), int(x / a)]

def plot_mesh_3d_flat(mesh, ax = None,
                      file_name = None, **kwargs):

    default_kwargs = {"label_cells": False,
                      "blocking"    : False # Defualt to non-blokcig behavior for plotting
                      }
    kwargs = {**default_kwargs, **kwargs}

    if ax:
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots()

    [Lx, Ly] = mesh.Ls[:]
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])

    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy] = [x1 - x0, y1 - y0]
            [cx, cy] = [(x0 + x1) / 2., (y0 + y1) / 2.]
            
            rect = Rectangle((x0, y0), dx, dy,
                             fill = False,
                             edgecolor = "black")
            ax.add_patch(rect)

            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, th1] = cell.pos[:]
                    [deg0, deg1] = [th0 * 180. / np.pi, th1 * 180. / np.pi]

                    wed = Wedge((cx, cy), min(dx, dy)/2, deg0, deg1,
                                fill = False,
                                edgecolor = "black"
                                )
                    ax.add_patch(wed)
                    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    if file_name:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.tight_layout()
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)

    return [fig, ax]
