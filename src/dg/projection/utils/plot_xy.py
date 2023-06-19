import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

sys.path.append('../..')
import dg.quadrature as qd
from dg.projection import push_forward, pull_back

def plot_xy(mesh, proj, file_name = None, **kwargs):
    
    default_kwargs = {'cmap' : 'hot'}
    kwargs = {**default_kwargs, **kwargs}
    
    if not mesh.has_th:
        return None
    
    [Lx, Ly] = mesh.Ls[:]
    Lth = 2. * np.pi

    # Integrate each cell in th, and get the bounds for the colorbar
    [vmin, vmax] = [0., 0.]
    col_items = sorted(mesh.cols.items())
    col_intg_ths = {}
    for col_key, col in col_items:
        if col.is_lf:
            [ndof_x, ndof_y] = col.ndofs[:]
            
            col_intg_th = np.zeros([ndof_x, ndof_y])
            
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [ndof_th]    = cell.ndofs[:]
                    [th_0, th_1] = cell.pos[:]
                    dth          = th_1 - th_0

                    [_, _, _, _, _, wth] = qd.quad_xyth(nnodes_th = ndof_th)
                    
                    proj_cell = proj.cols[col_key].cells[cell_key]
                    
                    for aa in range(0, ndof_th):
                        col_intg_th += (dth / 2.) * wth[aa] * proj_cell.vals[:, :, aa]
                        
            col_intg_ths[col_key] = col_intg_th

            vmin = min(np.amin(col_intg_th), vmin)
            vmax = max(np.amax(col_intg_th), vmax)
    
    fig, ax = plt.subplots()

    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy]         = [x1 - x0, y1 - y0]
            [ndof_x, ndof_y] = col.ndofs[:]
            
            [xxb, _, yyb, _, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                  nnodes_y = ndof_y)
            
            xxf = push_forward(x0, x1, xxb)
            yyf = push_forward(y0, y1, yyb)
            
            vals = col_intg_ths[col_key]
            
            pc = ax.pcolormesh(xxf, yyf, vals.transpose(),
                               cmap = kwargs['cmap'],
                               vmin = vmin, vmax = vmax,
                               shading = 'gouraud')
            
            rect = Rectangle((x0, y0), dx, dy, fill = False)
            ax.add_patch(rect)
    
    fig.colorbar(pc)

    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    
    if file_name:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)
        
    return [fig, ax]
