import os, sys
src_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, os.pardir, "src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Standard Library Imports
from time import perf_counter

# Third-Party Library Imports

# Local Library Imports
import dg.mesh  as ji_mesh
import dg.mesh.utils
import utils

def gen_mesh_plot(mesh, trial, trial_dir, **kwargs):
    
    default_kwargs = {'blocking' : False, # Default to non-blocking behavior for plotting
                      'verbose'  : False}
    kwargs = {**default_kwargs, **kwargs}

    if kwargs["verbose"]:
        perf_0 = perf_counter()
        msg = ( '[Trial {}] Plotting mesh...\n'.format(trial) )
        utils.print_msg(msg, **kwargs)
    
    file_name = 'mesh_3d_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    ji_mesh.utils.plot_mesh(mesh      = mesh,
                            file_name = file_path,
                            plot_dim  = 3)
    
    file_name = 'mesh_2d_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    ji_mesh.utils.plot_mesh(mesh        = mesh,
                            file_name   = file_path,
                            plot_dim    = 2,
                            label_cells = (trial <= 2))
    
    if kwargs["verbose"]:
        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        msg = ( '[Trial {}] Mesh plotted!\n'.format(trial) +
                22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff) )
        utils.print_msg(msg, **kwargs)

def gen_mesh_plot_p(mesh, trial, trial_dir, **kwargs):

    default_kwargs = {'blocking' : False, # Default to non-blocking behavior for plotting
                      "verbose"  : False}
    kwargs = {**default_kwargs, **kwargs}
    
    if kwargs["verbose"]:
        perf_0 = perf_counter()
        msg = ( '[Trial {}] Plotting mesh polynomial degree...\n'.format(trial) )
        utils.print_msg(msg, **kwargs)
    
    file_name = 'mesh_3d_p_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    ji_mesh.utils.plot_mesh_p(mesh        = mesh,
                              file_name   = file_path,
                              plot_dim    = 3)
    
    file_name = 'mesh_2d_p_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    ji_mesh.utils.plot_mesh_p(mesh        = mesh,
                              file_name   = file_path,
                              plot_dim    = 2,
                              label_cells = (trial <= 3))
    
    if kwargs["verbose"]:
        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        msg = ( '[Trial {}] Mesh polynomial degree plotted!\n'.format(trial) +
                22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff) )
        utils.print_msg(msg, **kwargs)