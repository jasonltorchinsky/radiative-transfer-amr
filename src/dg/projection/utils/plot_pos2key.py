import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

import numpy as np

from .plot_mesh import plot_mesh
from ..ji_mesh import get_key, pos2key

def plot_pos2key(mesh, npoints, file_name = None):

    [fig, ax] = plot_mesh(mesh)

    # Pick out npoints random points
    Lx = mesh.Ls[0]
    Ly = mesh.Ls[1]
    rng = np.random.default_rng()
    colors = ['black', 'blue', 'red', 'green', 'cyan', 'magenta', 'yellow']
    ncolors = len(colors)
    for ii in range(0, npoints):
        color = colors[np.mod(ii, ncolors)]
        # Put point on plot and color cell that it's in
        [xp, yp] = rng.random([2])
        circ  = Circle((xp, yp), radius = Lx/100,
                       color = color, fill = True)
        
        key = pos2key(mesh, [xp, yp])
        [x0, y0, x1, y1] = mesh.pos[key]
        width = x1 - x0
        height = y1 - y0

        rect = Rectangle((x0, y0), width, height,
                         color = color, fill = True, alpha = 0.1)
        ax.add_patch(rect)
        ax.add_patch(circ)

    
    if file_name:
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)

    return ax
