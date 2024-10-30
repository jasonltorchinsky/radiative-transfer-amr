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

        ## Navigate each trial directory path
        subdir_paths: list = [subdir_path 
                              for subdir_path in os.listdir(ref_strat_dir_path)
                              if os.path.isdir(os.path.join(ref_strat_dir_path, subdir_path))]
        trial_dir_paths: list = [os.path.join(ref_strat_dir_path, subdir_path) 
                                 for subdir_path in np.sort(np.array(subdir_paths, dtype = consts.INT)).astype(str)
                                 if os.path.isdir(os.path.join(ref_strat_dir_path, subdir_path))]
        for trial_dir_path in trial_dir_paths:
            ## Plot the convergence information for the linear solves
            convergence_info_file_name: str = "convergence_info.json"
            convergence_info_file_path: str = os.path.join(trial_dir_path,
                                                           convergence_info_file_name)
            
            convergence_info_plot_file_name: str = "convergence_info.png"
            convergence_info_plot_file_path: str = os.path.join(trial_dir_path,
                                                                convergence_info_plot_file_name)
            
            if (os.path.isfile(convergence_info_file_path)
                and not os.path.isfile(convergence_info_plot_file_path)):
                with open(convergence_info_file_path, "r") as convergence_info_file:
                    convergence_info: dict = json.load(convergence_info_file)
                plot_convergence_info(convergence_info, convergence_info_plot_file_path)

            convergence_info_hr_file_name: str = "convergence_info_hr.json"
            convergence_info_hr_file_path: str = os.path.join(trial_dir_path,
                                                           convergence_info_hr_file_name)
            
            convergence_info_hr_plot_file_name: str = "convergence_info_hr.png"
            convergence_info_hr_plot_file_path: str = os.path.join(trial_dir_path,
                                                                convergence_info_hr_plot_file_name)
            
            if (os.path.isfile(convergence_info_hr_file_path)
                and not os.path.isfile(convergence_info_hr_plot_file_path)):
                with open(convergence_info_hr_file_path, "r") as convergence_info_hr_file:
                    convergence_info_hr: dict = json.load(convergence_info_hr_file)
                plot_convergence_info(convergence_info_hr, convergence_info_hr_plot_file_path)

def plot_convergence_info(convergence_info: dict, file_path: str) -> None:
    fig, ax = plt.subplots()
    
    ## Plot the residuals
    iters: np.ndarray = np.arange(1, convergence_info["iteration_number"] + 2, dtype = consts.INT)
    residuals: np.ndarray = np.array(convergence_info["convergence_history"])
    ax.plot(iters, residuals,
            color = "black",
            marker = "o",
            linestyle = "--")
    
    ## Plot the tolerances
    plt.axhline(y = convergence_info["atol"],
                xmin = 0., xmax = 1.,
                color = "red",
                linestyle = ":",
                label = "atol")
    
    if convergence_info["iteration_number"] == convergence_info["max_it"]:
        plt.axvline(x = convergence_info["max_it"],
                    ymin = 0., ymax = 1.,
                    color = "gray",
                    linestyle = "--",
                    label = "max_it")
        
    ax.legend()

    xmin: float = 0.
    xmax: float = convergence_info["iteration_number"]
    ax.set_xlim([xmin, xmax + 1])

    ax.set_yscale("log", base = 10)
    ymin: float = min(convergence_info["atol"], convergence_info["res_best"])
    ymax: float = np.max(residuals)
    ax.set_ylim([0.5 * ymin, 1.5 * ymax])
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual")
        
    title: str = "Converged Reason: {}".format(convergence_info["converged_reason"])
    ax.set_title(title)
    
    fig.set_size_inches(6.5, 6.5)
    plt.tight_layout()
    plt.savefig(file_path, dpi = 300)
    plt.close(fig)


if __name__ == "__main__":
    main()