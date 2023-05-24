import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

sys.path.append('../..')
from dg.quadrature import quad_xyth, lag_eval
from dg.projection import push_forward, pull_back

def plot_angular_dists(mesh, proj, file_name = None, **kwargs):

    if not mesh.has_th:
        return None
    
    [Lx, Ly] = mesh.Ls[:]
    Lth = 2. * np.pi

    # Integrate each cell in x, and get the bounds for the colorbar
    [vmin, vmax] = [0., 0.]
    col_items = sorted(mesh.cols.items())
    col_lfs = ((col_key, col) for col_key, col in col_items if col.is_lf)
    cell_intg_xs = {}
    for col_key, col in col_lfs:
        [ndof_x, ndof_y] = col.ndofs[:]

        [_, wx, _, _, _, _] = qd.quad_xyth(nnodes_x = ndof_x)

        cell_items = sorted(col.cells.items())
        cell_lfs = ((cell_key, cell) for cell_key, cell in cell_items if cell.is_lf)

        for cell_key, cell in cell_lfs:
            [ndof_th] = cell.ndofs[:]
            proj_cell = proj.cols[col_key].cells[cell_key]

            cell_intg_x = np.zeros([ndof_y, ndof_th])
            for ii in range(0, ndof_x):
                cell_intg_x += wx[ii] * proj_cell[ii, :, :]

            cell_intg_xs[(col_key, cell_key)] = cell_intg_x
    
    fig, axs = plt.subplots()

    for col_key, col in col_lfs:
        if col.is_lf:
            [_, y0, _, y1] = col.pos[:]
            dy             = y1 - y0
            [_, ndof_y]    = col.ndofs[:]
            
            [_, _, yyb, _, _, _] = quad_xyth(nnodes_y = ndof_y)
            
            yyf = push_forward(y0, y1, yyb)
        
            cell_items = sorted(col.cells.items())
            cell_lfs = ((cell_key, cell) for cell_key, cell in cell_items if cell.is_lf)

            for cell_key, cell in cell_lfs:
                [th0, th1] = cell.pos[:]
                dth        = th1 - th0
                [ndof_th]  = cell.ndofs[:]
                [_, _, _, _, thb, _] = quad_xyth(nnodes_th = ndof_th)
                
                thf = push_forward(th0, th1, thb)
                vals = cell_intg_xs[(col_key, cell_key)]
                
                pc = ax.pcolormesh(yyf, thf, vals.transpose(),
                                   cmap = kwargs['cmap'],
                                   vmin = vmin, vmax = vmax,
                                   shading = 'gouraud')
                    
                rect = Rectangle((th0, y0), dth, dy, fill = False)
            ax.add_patch(rect)

        nth_ticks = 9
        th_ticks = np.linspace(0, 2, nth_ticks) * np.pi
        th_tick_labels = [None] * nth_ticks
        for aa in range(0, nth_ticks):
            th_rad = th_ticks[aa] / np.pi
            th_tick_labels[aa] = '{:.2f}\u03C0'.format(th_rad)
        ax.set_xticks(th_ticks)
        ax.set_xticklabels(th_tick_labels)
    
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
