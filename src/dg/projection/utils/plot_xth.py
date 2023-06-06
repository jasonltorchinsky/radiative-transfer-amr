import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

sys.path.append('../..')
import dg.quadrature as qd
from dg.projection import push_forward, pull_back

def plot_xth(mesh, proj, file_name = None, **kwargs):
    
    default_kwargs = {'cmap' : 'hot'}
    kwargs = {**default_kwargs, **kwargs}
    
    if not mesh.has_th:
        return None
    
    [Lx, Ly] = mesh.Ls[:]
    Lth = 2. * np.pi

    # Integrate each cell in x, and get the bounds for the colorbar
    [vmin, vmax] = [0., 0.]
    col_items = sorted(mesh.cols.items())
    cell_intg_ys = {}
    for col_key, col in col_items:
        if col.is_lf:
            [ndof_x, ndof_y] = col.ndofs[:]
            
            [_, _, _, wy, _, _] = qd.quad_xyth(nnodes_y = ndof_y)
            
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [ndof_th] = cell.ndofs[:]
                    proj_cell = proj.cols[col_key].cells[cell_key]
                    
                    cell_intg_y = np.zeros([ndof_x, ndof_th])
                    for jj in range(0, ndof_y):
                        cell_intg_y += wy[jj] * proj_cell.vals[:, jj, :]
                        
                    cell_intg_ys[(col_key, cell_key)] = cell_intg_y

                    vmin = min(np.amin(cell_intg_y), vmin)
                    vmax = max(np.amax(cell_intg_y), vmax)
    
    fig, ax = plt.subplots()

    for col_key, col in col_items:
        if col.is_lf:
            [x0, _, x1, _] = col.pos[:]
            dx             = x1 - x0
            [ndof_x, _]    = col.ndofs[:]
            
            [xxb, _, _, _, _, _] = qd.quad_xyth(nnodes_x = ndof_x)
            
            xxf = push_forward(x0, x1, xxb)
            
            cell_items = sorted(col.cells.items())
            
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, th1] = cell.pos[:]
                    dth        = th1 - th0
                    [ndof_th]  = cell.ndofs[:]
                    [_, _, _, _, thb, _] = qd.quad_xyth(nnodes_th = ndof_th)
                    
                    thf = push_forward(th0, th1, thb)
                    vals = cell_intg_ys[(col_key, cell_key)]
                    
                    pc = ax.pcolormesh(thf, xxf, vals,
                                       cmap = kwargs['cmap'],
                                       vmin = vmin, vmax = vmax,
                                       shading = 'gouraud')
                    
                    rect = Rectangle((th0, x0), dth, dx, fill = False)
                    ax.add_patch(rect)
    
    fig.colorbar(pc)
    
    nth_ticks = 9
    th_ticks = np.linspace(0, 2, nth_ticks) * np.pi
    th_tick_labels = [None] * nth_ticks
    for aa in range(0, nth_ticks):
        th_rad = th_ticks[aa] / np.pi
        th_tick_labels[aa] = '{:.2f}\u03C0'.format(th_rad)
    ax.set_xticks(th_ticks)
    ax.set_xticklabels(th_tick_labels)

    ax.set_xlim([0, Lth])
    ax.set_ylim([0, Lx])
    
    if file_name:
        fig.set_size_inches(6.5, 6.5 * (Lx / Lth))
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)
        
    return [fig, ax]