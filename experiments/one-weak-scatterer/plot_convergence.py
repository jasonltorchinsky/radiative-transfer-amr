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
from dg.mesh import Mesh
from dg.mesh import from_file as mesh_from_file

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

        if os.path.isfile(tracked_values_file_path):
            with open(tracked_values_file_path) as tracked_values_file:
                ref_strat_err_dict: dict = json.load(tracked_values_file)
        else:
            prev_ndof: int = 1

            dir_walks = os.walk(ref_strat_dir)
            next(dir_walks) # Skip the first one
            for dir_walk in dir_walks:
                trial_dir: str = dir_walk[0]
                ## Read the mesh from file to get ndof
                mesh_file_name: str = "mesh.json"
                mesh_file_path: str = os.path.join(trial_dir, mesh_file_name)

                mesh: Mesh = mesh_from_file(mesh_file_path)
                ndof: int = mesh.get_ndof()

                if (float(ndof) / float(prev_ndof) >= 1.25):
                    err_ind_file_name: str = "err_ind_anl.json"
                    err_ind_file_path: str = os.path.join(trial_dir, err_ind_file_name)

                    err_ind_file = open(err_ind_file_path)
                    error: float = json.load(err_ind_file)["mesh_error"]

                    ref_strat_err_dict[ndof] = error

                    prev_ndof: int = ndof

            with open(tracked_values_file_path, "w") as tracked_values_file:
                json.dump(ref_strat_err_dict, tracked_values_file)
        
        tracked_values[ref_strat_name] = ref_strat_err_dict

    ## Plot the error for each refinement strategy
    fig, ax = plt.subplots()
    
    colors: list = ["#000000", "#E69F00", "#56B4E9", "#009E73",
                    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
                    "#882255"]
    markers: list = [".", "v", "s", "*", "^", "D", "P", "X"]
    style_idx: int = 0
    
    for ref_strat_name, err_dict in tracked_values.items():
        ndofs: np.ndarray  = np.array(list(err_dict.keys()), dtype = consts.INT)
        errors: np.ndarray = np.array(list(err_dict.values()), dtype = consts.INT)

        ax.plot(ndofs, errors,
                label = ref_strat_name,
                color = colors[style_idx%len(colors)],
                marker = markers[style_idx%len(markers)],
                linestyle = "--")
        
        style_idx += 1
        
    ax.legend()
    
    ax.set_yscale("log", base = 10)
    
    ax.set_xlabel("Total Degrees of Freedom")
    ax.set_ylabel(r"$Error := \sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u - u_{hp} \right)^2\,d\vec{x}\,d\vec{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\vec{s}}}$")
    #ax.set_ylabel(r"$Error := \sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} - u_{hp} \right)^2\,d\vec{x}\,d\vec{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} \right)^2\,d\vec{x}\,d\vec{s}}}$")
        
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