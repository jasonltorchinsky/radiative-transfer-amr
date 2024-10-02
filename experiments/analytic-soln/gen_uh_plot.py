import os, sys
src_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, os.pardir, "src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Standard Library Imports
from time import perf_counter

# Third-Party Library Imports

# Local Library Imports
import dg.projection as proj
import dg.projection.utils
import utils

    
def gen_uh_plot(mesh, uh_proj, trial, trial_dir, **kwargs):
    
    default_kwargs = {"blocking" : False, # Default to non-blocking behavior for plotting
                      "verbose"  : False}
    kwargs = {**default_kwargs, **kwargs}
    
    if kwargs["verbose"]:
        perf_0: float = perf_counter()
        msg: str = ( "[Trial {}] Plotting numerical solution...\n".format(trial) )
        utils.print_msg(msg, **kwargs)
    
    file_name = "uh_th_{}.png".format(trial)
    file_path = os.path.join(trial_dir, file_name)
    proj.utils.plot_th(mesh, uh_proj, file_name = file_path)
    
    file_name = "uh_xy_{}.png".format(trial)
    file_path = os.path.join(trial_dir, file_name)
    proj.utils.plot_xy(mesh, uh_proj, file_name = file_path)
    
    file_name = "uh_xth_{}.png".format(trial)
    file_path = os.path.join(trial_dir, file_name)
    proj.utils.plot_xth(mesh, uh_proj, file_name = file_path)
    
    file_name = "uh_yth_{}.png".format(trial)
    file_path = os.path.join(trial_dir, file_name)
    proj.utils.plot_yth(mesh, uh_proj, file_name = file_path)
    
    file_name = "uh_xyth_{}.png".format(trial)
    file_path = os.path.join(trial_dir, file_name)
    proj.utils.plot_xyth(mesh, uh_proj, file_name = file_path)
    
    if kwargs["verbose"]:
        perf_f: str = perf_counter()
        perf_diff: str = perf_f - perf_0
        msg: str = ( "[Trial {}] Numerical solution plotted!\n".format(trial) +
                     22 * " " + "Time Elapsed: {:08.3f} [s]\n".format(perf_diff) )
        utils.print_msg(msg, **kwargs)