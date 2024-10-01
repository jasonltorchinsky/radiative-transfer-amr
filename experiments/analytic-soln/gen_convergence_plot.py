import os, sys
src_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, os.pardir, "src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Standard Library Imports
from time import perf_counter

# Third-Party Library Imports
import matplotlib.pyplot as plt
import numpy as np

# Local Library Imports
import utils

    
def gen_convergence_plot(ref_strats, ref_strat_names, ref_strat_ndofs,
                         ref_strat_errs, out_dir, **kwargs):
    
    default_kwargs = {'blocking' : False, # Default to non-blocking behavior for plotting
                      "verbose"  : False}
    kwargs = {**default_kwargs, **kwargs}
    
    if kwargs["verbose"]:
        perf_0: float = perf_counter()
        msg: str = ( 'Plotting convergence rates...\n' )
        utils.print_msg(msg, **kwargs)
    
    fig, ax = plt.subplots()
            
    nref_strat = len(ref_strats)
            
    colors = ['#000000', '#E69F00', '#56B4E9', '#009E73',
              '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
              '#882255']
            
    for cc in range(0, nref_strat):
        ref_strat_name = ref_strat_names[cc]
        ndofs = ref_strat_ndofs[ref_strat_name]
        errs  = ref_strat_errs[ref_strat_name]
        ax.scatter(ndofs, errs,
                   label     = None,
                   color     = colors[cc])
                
        # Get best-fit line
        [a, b] = np.polyfit(np.log10(ndofs), np.log10(errs), 1)
        xx = np.logspace(np.log10(ndofs[0]), np.log10(ndofs[-1]))
        yy = 10**b * xx**a
        ax.plot(xx, yy,
                label = '{}: {:4.2f}'.format(ref_strat_name, a),
                color = colors[cc],
                linestyle = '--'
                )
                
    ax.legend()
            
    ax.set_xscale('log', base = 10)
    ax.set_yscale('log', base = 10)
            
    ax.set_xlabel('Total Degrees of Freedom')
    ax.set_ylabel(r'$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u - u_{hp} \right)^2\,d\vec{x}\,d\vec{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\vec{s}}}$')

    title_str = ( 'Convergence Rate' )
    ax.set_title(title_str)
            
    file_name = 'convergence.png'
    file_path = os.path.join(out_dir, file_name)
    fig.set_size_inches(6.5, 6.5)
    plt.tight_layout()
    plt.savefig(file_path, dpi = 300)
    plt.close(fig)
    
    if kwargs["verbose"]:
        perf_f: str = perf_counter()
        perf_diff: str = perf_f - perf_0
        msg: str = ( 'Convergence rates plotted!\n' +
                     22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff) )
        utils.print_msg(msg, **kwargs)