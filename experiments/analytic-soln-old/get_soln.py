import os, sys
src_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, os.pardir, "src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Standard Library Imports
from time import perf_counter

# Third-Party Library Imports
from petsc4py import PETSc

# Local Library Imports
import rt
import utils


def get_soln(mesh, kappa, sigma, Phi, bcs_dirac, f, trial, **kwargs):
    
    default_kwargs = {"verbose"  : False}
    kwargs = {**default_kwargs, **kwargs}
    
    if kwargs["verbose"]:
        perf_0: float = perf_counter()
        msg: str = ( "[Trial {}] Obtaining numerical solution...\n".format(trial) )
        utils.print_msg(msg)
    
    [uh_proj, info, mat_info] = rt.rtdg(mesh, kappa, sigma, Phi, bcs_dirac, f,
                                        verbose = True)
    PETSc.garbage_cleanup()
    
    if kwargs["verbose"]:
        perf_f: float = perf_counter()
        perf_diff: float = perf_f - perf_0
        msg: str = ( "[Trial {}] Numerical solution obtained! : ".format(trial) +
            "Exit Code {} \n".format(info) +
            22 * " " + "Time Elapsed: {:08.3f} [s]\n".format(perf_diff) )
        utils.print_msg(msg)

    return [uh_proj, info, mat_info]