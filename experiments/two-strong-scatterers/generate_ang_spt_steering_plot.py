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


    ## Navigate each refinement strategy directory
    for ref_strat_name, _ in refinement_strategies.items():
        ref_strat_dir_path: str = os.path.join(out_dir_path, ref_strat_name)

        
        ## Plot the convergence information for the linear solves
        ref_strat_log_file_name: str = "ref_strat_log.json"
        ref_strat_log_file_path: str = os.path.join(ref_strat_dir_path,
                                                    ref_strat_log_file_name)
        
        ang_spt_steering_plot_file_name: str = "ang_spt_steering.png"
        ang_spt_steering_plot_file_path: str = os.path.join(ref_strat_dir_path,
                                                            ang_spt_steering_plot_file_name)
        
        if (os.path.isfile(ref_strat_log_file_path)
            and not os.path.isfile(ang_spt_steering_plot_file_path)):
            with open(ref_strat_log_file_path, "r") as ref_strat_log_file:
                ref_strat_log: dict = json.load(ref_strat_log_file)
            plot_ang_spt_steering(ref_strat_log, ang_spt_steering_plot_file_path)


def plot_ang_spt_steering(ref_strat_log: dict, file_path: str) -> None:
    fig, ax = plt.subplots()
    
    ## Obtain a list of all refinement kinds
    size: int = 1000
    trials: np.ndarray = np.zeros(size, dtype = consts.INT)
    ref_kinds: np.ndarray = np.zeros(size, dtype = consts.INT)
    idx: int = 0
    for key in ref_strat_log.keys():
        if key.isnumeric():
            trial: int = int(key)
            ref_kind: str = ref_strat_log[key]["ref_kind"]

            trials[idx] = trial
            if (ref_kind == "ang"):
                ref_kinds[idx] = 1
            else:
                ref_kinds[idx] = -1
            
            idx += 1

    ## Plot the residuals
    ax.scatter(trials[:idx], ref_kinds[:idx],
               color = "black",
               marker = "o")
    
    ## Set the y-ticks to label the refinement strategies
    ax.set_yticks([-1., 1.], ["Spt.", "Ang."])
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Refinement Kind")
    
    fig.set_size_inches(6.5, 6.5)
    plt.tight_layout()
    plt.savefig(file_path, dpi = 300)
    plt.close(fig)


if __name__ == "__main__":
    main()