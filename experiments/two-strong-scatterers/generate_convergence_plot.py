import os, sys
src_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, os.pardir, "src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Standard Library Imports
import argparse
import json

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt

# Local Library Imports
import consts

# Relative Imports
from refinement_strategies import refinement_strategies

def main():
    ## Read command-line input
    parser_desc: str = ( "Plots the convergence rates for each refinement strategy." )
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description = parser_desc)
    
    parser.add_argument("--o",
                        action = "store",
                        nargs = 1, 
                        type = str,
                        required = False,
                        default = "out",
                        help = "Output directory path.")
    
    args = parser.parse_args()

    if (args.o != "out"):
        out_dir_path: str = os.path.normpath(args.o[0])
    else:
        out_dir_path: str = args.o

    ## Read tracked values from each strategy
    tracked_values: dict = {}
    for ref_strat_name, _ in refinement_strategies.items():
        ref_strat_dir: str = os.path.join(out_dir_path, ref_strat_name)

        tracked_values_file_name: str = "tracked_vals.json"
        tracked_values_file_path: str = os.path.join(ref_strat_dir,
                                                     tracked_values_file_name)

        with open(tracked_values_file_path, "r") as tracked_values_file:
            ref_strat_err_dict: dict = json.load(tracked_values_file)
        
        tracked_values[ref_strat_name] = ref_strat_err_dict

    ## Plot the error for each refinement strategy
    fig, ax = plt.subplots()
    
    colors: list = ["#000000", "#E69F00", "#56B4E9", "#009E73",
                    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
                    "#882255"]
    markers: list = [".", "v", "s", "*", "^", "D", "P", "X"]
    style_idx: int = 0

    ref_strat_labels: dict = {"hp-amr-spt-p-uni-ang" : r"$hp$-Adap. Spt., $p$-Unif. Ang.",
                              "hp-amr-ang-p-uni-spt" : r"$hp$-Adap. Ang., $p$-Unif. Spt.",
                              "hp-amr-ang-hp-amr-spt" : r"$hp$-Adap. Ang., $hp$-Adap. Spt."}
    
    for ref_strat_name, err_dict in tracked_values.items():
        ndofs: np.ndarray  = np.array(list(err_dict.keys()), dtype = consts.INT)
        errors: np.ndarray = np.array(list(err_dict.values()))

        ax.plot(ndofs, errors,
                label = ref_strat_labels[ref_strat_name],
                color = colors[style_idx%len(colors)],
                marker = markers[style_idx%len(markers)],
                linestyle = "--")
        
        style_idx += 1
        
    ax.legend()
    
    ax.set_yscale("log", base = 10)
    ax.set_ylim([1.e-3, 0.3])
    
    ax.set_xlabel("Total Degrees of Freedom")
    ax.set_ylabel(r"$Error := \sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} - u_{hp} \right)^2\,d\vec{x}\,d\vec{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} \right)^2\,d\vec{x}\,d\vec{s}}}$")
        
    title: str = ( "Convergence Rate" )
    ax.set_title(title)
    
    file_name: str = "convergence.png"
    file_path: str = os.path.join(out_dir_path, file_name)
    fig.set_size_inches(6.5, 6.5)
    plt.tight_layout()
    plt.savefig(file_path, dpi = 300)
    plt.close(fig)

if __name__ == "__main__":
    main()