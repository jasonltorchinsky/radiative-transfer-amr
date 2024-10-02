import os, sys
src_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, os.pardir, "src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Standard Library Imports
from time import perf_counter

# Third-Party Library Imports

# Local Library Imports
import amr
import amr.utils
import utils
    
def gen_err_ind_plot(mesh, err_ind, trial, trial_dir, file_name, **kwargs):
    
    default_kwargs = {"blocking" : False, # Default to non-blocking behavior for plotting
                      "verbose"  : False}
    kwargs = {**default_kwargs, **kwargs}
    
    if kwargs["verbose"]:
        perf_0: str = perf_counter()
        msg: str = ( "[Trial {}] Plotting error indicators...\n".format(trial) )
        utils.print_msg(msg, **kwargs)
    
    file_path = os.path.join(trial_dir, file_name)
    amr.utils.plot_error_indicator(mesh, err_ind, file_name = file_path)
    if err_ind.ref_cell:
        file_path = os.path.join(trial_dir, "cell_jumps.png")
        amr.utils.plot_cell_jumps(mesh, err_ind, file_name = file_path)
    
    if kwargs["verbose"]:
        perf_f: float = perf_counter()
        perf_diff: float = perf_f - perf_0
        msg: str = ( "[Trial {}] Error indicator plotted!\n".format(trial) +
                     22 * " " + "Time Elapsed: {:08.3f} [s]\n".format(perf_diff) )
        utils.print_msg(msg, **kwargs)