import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys

sys.path.append('../..')
from dg.quadrature import quad_xyth, lag_eval
from dg.projection import push_forward, pull_back

def plot_error_indicator_dist(mesh, err_ind, file_name = None, **kwargs):
    
    default_kwargs = {'name' : ''}
    kwargs = {**default_kwargs, **kwargs}

    if err_ind.by_col:
        [fig, ax] = plot_error_indicator_by_column(mesh,
                                                   err_ind,
                                                   file_name = file_name,
                                                   **kwargs)
    if err_ind.by_cell:
        [fig, ax] = plot_error_indicator_by_cell(mesh,
                                                 err_ind,
                                                 file_name = file_name,
                                                 **kwargs)
        
    return [fig, ax]

def plot_error_indicator_by_column(mesh, err_ind, file_name = None, **kwargs):

    default_kwargs = {}
    kwargs = {**default_kwargs, **kwargs}
    
    col_items = sorted(mesh.cols.items())
    err_ind_dict = {}
    for col_key, col in col_items:
        if col.is_lf:
            err_ind_dict[col_key] = err_ind.cols[col_key].err_ind
    
    err_ind_vals = err_ind_dict.values()
    
    fig, ax = plt.subplots()
    
    ax.boxplot(err_ind_vals,
               vert = False,
               whis = [0, 90])
    
    ax.tick_params(
        axis      = 'y',    # changes apply to the y-axis
        which     = 'both', # both major and minor ticks are affected
        left      = False,  # ticks along the bottom edge are off
        right     = False,  # ticks along the top edge are off
        labelleft = False)  # labels along the bottom edge are off
    
    ax.set_xscale('log', base = 2)
    
    xmin = 2**(np.floor(np.log2(min(err_ind_vals))))
    xmax = 2**(np.ceil(np.log2(max(err_ind_vals))))
    ax.set_xlim([xmin, xmax])

    yy = np.random.normal(1, 0.04, size = len(err_ind_vals))
    ax.plot(err_ind_vals, yy, 'k.', alpha = 0.8)
    
    title_str = kwargs['name'] + ' Error Indicator'
    ax.set_title(title_str)
    
    if file_name:
        fig.set_size_inches(6, 6)
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)

    return [fig, ax]

def plot_error_indicator_by_cell(mesh, err_ind, file_name = None, **kwargs):

    default_kwargs = {}
    kwargs = {**default_kwargs, **kwargs}
    
    col_items = sorted(mesh.cols.items())
    err_ind_dict = {}
    for col_key, col in col_items:
        if col.is_lf:
            cell_items = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                err_ind_dict[(col_key, cell_key)] = \
                    err_ind.cols[col_key].cells[cell_key].err_ind
    
    err_ind_vals = err_ind_dict.values()
    
    fig, ax = plt.subplots()
    
    ax.boxplot(err_ind_vals,
               vert = False,
               whis = [0, 90])
    
    ax.tick_params(
        axis      = 'y',    # changes apply to the y-axis
        which     = 'both', # both major and minor ticks are affected
        left      = False,  # ticks along the bottom edge are off
        right     = False,  # ticks along the top edge are off
        labelleft = False)  # labels along the bottom edge are off
    
    ax.set_xscale('log', base = 2)
    
    xmin = 2**(np.floor(np.log2(min(err_ind_vals))))
    xmax = 2**(np.ceil(np.log2(max(err_ind_vals))))
    ax.set_xlim([xmin, xmax])

    yy = np.random.normal(1, 0.04, size = len(err_ind_vals))
    ax.plot(err_ind_vals, yy, 'k.', alpha = 0.8)
    
    title_str = kwargs['name'] + ' Error Indicator'
    ax.set_title(title_str)
    
    if file_name:
        fig.set_size_inches(6, 6)
        plt.savefig(file_name, dpi = 300)
        plt.close(fig)

    return [fig, ax]
