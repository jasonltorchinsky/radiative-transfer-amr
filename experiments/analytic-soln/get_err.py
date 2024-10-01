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


def get_err(mesh, uh_proj, u, kappa, sigma, Phi, bcs_dirac, f,
            trial, figs_dir, **kwargs):
    
    default_kwargs = {'res_coeff' : 1,
                      'err_kind'  : 'anl',
                      "verbose"   : False}
    kwargs = {**default_kwargs, **kwargs}
    
    if kwargs["verbose"]:
        perf_0: float = perf_counter()
        msg: str = ( '[Trial {}] Obtaining error...\n'.format(trial) )
        utils.print_msg(msg)

    if kwargs['err_kind'] == 'anl':
        err = amr.total_anl_err(mesh, uh_proj, u, **kwargs)
    elif kwargs['err_kind'] == 'hr':
        err = amr.high_res_err(mesh, uh_proj,
                               kappa, sigma, Phi, bcs_dirac, f,
                               verbose = True,
                               dir_name = figs_dir,
                               **kwargs)
    
    if kwargs["verbose"]:
        perf_f = perf_counter()
        perf_diff = perf_f - perf_0
        msg = ( '[Trial {}] Error obtained! : {:.4E}\n'.format(trial, err) +
            22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff) )
        utils.print_msg(msg)
    
    return err