import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge
from matplotlib.collections import PatchCollection
import sys

sys.path.append("../..")
from dg.quadrature import quad_xyth, lag_eval
from dg.projection import push_forward, pull_back

def plot_error_indicator(mesh, err_ind, file_name = None, **kwargs):
    
    default_kwargs = {"cmap" : "Reds",
                      "name" : "",
                      "ref_col"  : err_ind.ref_col,
                      "ref_cell" : err_ind.ref_cell}
    kwargs = {**default_kwargs, **kwargs}

    if err_ind.ref_col and kwargs["ref_col"]:
        [fig, ax] = plot_error_indicator_by_column(mesh,
                                                   err_ind,
                                                   file_name = file_name,
                                                   **kwargs)
    elif err_ind.ref_cell and kwargs["ref_cell"]:
        [fig, ax] = plot_error_indicator_by_cell(mesh,
                                                 err_ind,
                                                 file_name = file_name,
                                                 **kwargs)
        
    return [fig, ax]

def plot_error_indicator_by_column(mesh, err_ind, file_name = None, **kwargs):

    default_kwargs = {}
    kwargs = {**default_kwargs, **kwargs}
    
    fig, ax = plt.subplots()
    
    cmap = cm.get_cmap(kwargs["cmap"])
    
    [Lx, Ly] = mesh.Ls[:]
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    
    # Get colorbar min/max
    col_items = sorted(mesh.cols.items())
    if err_ind.col_max_err > 0:
        [vmin, vmax] = [10.**10, err_ind.col_max_err]
        for col_key, col in col_items:
            if col.is_lf:
                vmin = min(vmin, err_ind.cols[col_key].err)
    else:
        [vmin, vmax] = [err_ind.col_max_err, -10.**10]
        cmap         = cmap.reversed()
        for col_key, col in col_items:
            if col.is_lf:
                vmax = max(vmax, err_ind.cols[col_key].err)   
    rects = []
    rect_colors = []
    for col_key, col in col_items:
        if col.is_lf:
            # Plot column err indicator
            [x0, y0, x1, y1] = col.pos
            [dx, dy] = [x1 - x0, y1 - y0]
            [cx, cy] = [(x0 + x1) / 2., (y0 + y1) / 2.]
            [ndof_x, ndof_y] = col.ndofs
            
            col_err = err_ind.cols[col_key].err
            
            rect = Rectangle((x0, y0), dx, dy,
                             facecolor = cmap(col_err),
                             edgecolor = "black")
            rects += [rect]
            rect_color = col_err
            rect_colors += [rect_color]
            
    rect_coll = PatchCollection(rects, edgecolor = "black",
                                cmap = cmap)
    rect_coll.set_array(rect_colors)
    rect_coll.set_clim([vmin, vmax])
    ax.add_collection(rect_coll)
    
    fig.colorbar(rect_coll, ax = ax)

    title_str = "Max Error: {:.4E} \nAvg. Ref. Error {:.4E}".format(err_ind.col_max_err,
                                                                    err_ind.avg_col_ref_err)
    ax.set_title(title_str)
    
    if file_name:
        fig.set_size_inches(12, 12 * (Ly / Lx))
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)

    return [fig, ax]

def plot_error_indicator_by_cell(mesh, err_ind, file_name = None, **kwargs):
    
    default_kwargs = {}
    kwargs = {**default_kwargs, **kwargs}
    
    cmap = cm.get_cmap(kwargs["cmap"])
    
    fig, ax = plt.subplots()
    
    [Lx, Ly] = mesh.Ls[:]
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    
    # Get colorbar min/max
    col_items = sorted(mesh.cols.items())
    if err_ind.cell_max_err > 0:
        [vmin, vmax] = [10.**10, err_ind.cell_max_err]
        for col_key, col in col_items:
            if col.is_lf:
                cell_items = sorted(col.cells.items())
                for cell_key, cell in cell_items:
                    if cell.is_lf:
                        vmin = min(vmin, err_ind.cols[col_key].cells[cell_key].err)
    else:
        [vmin, vmax] = [err_ind.cell_max_err, -10.**10]
        cmap         = cmap.reversed()
        for col_key, col in col_items:
            if col.is_lf:
                cell_items = sorted(col.cells.items())
                for cell_key, cell in cell_items:
                    if cell.is_lf:
                        vmax = max(vmax, err_ind.cols[col_key].cells[cell_key].err)

    wedges = []
    wedge_colors = []
    rects = []
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos[:]
            [dx, dy] = [x1 - x0, y1 - y0]
            [cx, cy] = [(x0 + x1) / 2., (y0 + y1) / 2.]
            
            rect = Rectangle((x0, y0), dx, dy,
                             facecolor = "none",
                             edgecolor = "black")
            rects += [rect]

            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [th0, th1] = cell.pos[:]
                    [deg0, deg1] = [th0 * 180. / np.pi, th1 * 180. / np.pi]
                    
                    cell_err = err_ind.cols[col_key].cells[cell_key].err
                    
                    wedge = Wedge((cx, cy), min(dx, dy)/2, deg0, deg1,
                                  facecolor = cmap(cell_err),
                                  edgecolor = "black")
                    wedges += [wedge]
                    wedge_color = cell_err
                    wedge_colors += [wedge_color]
                    
    rect_coll = PatchCollection(rects, match_original = True)
    ax.add_collection(rect_coll)
    
    wedge_coll = PatchCollection(wedges, edgecolor = "black", cmap = cmap)
    wedge_coll.set_array(wedge_colors)
    wedge_coll.set_clim([vmin, vmax])
    ax.add_collection(wedge_coll)
    
    fig.colorbar(wedge_coll, ax = ax)

    title_str = "Max Error: {:.4E} \nAvg. Ref. Error {:.4E}".format(err_ind.cell_max_err,
                                                                    err_ind.avg_cell_ref_err)
    ax.set_title(title_str)
    
    if file_name:
        fig.set_size_inches(6.5, 6.5 * (Ly / Lx))
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)

    return [fig, ax]
