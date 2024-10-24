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
import petsc4py
from mpi4py import MPI
from petsc4py import PETSc

# Local Library Imports
import consts
from amr.error_indicator import Error_Indicator
from dg.projection import Projection
from dg.projection import from_file as projection_from_file

# Relative Imports
from problem import u
from refinement_strategies import refinement_strategies

def main():
    ## Initialize parallel communicators
    mpi_comm: MPI.Intracomm = MPI.COMM_WORLD
    
    if not PETSc.Sys.isInitialized():
        petsc4py.init(comm = mpi_comm)
    petsc_comm: PETSc.Comm = PETSc.COMM_WORLD
    comm_rank: int = petsc_comm.getRank()

    ## Read command-line input
    parser_desc: str = ( "Calculates the error history for each refinement strategy.")
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

    ## Read input - hardcoded file names
    with open("input.json", "r") as input_file:
        input_dict: dict = json.load(input_file)

    ndof_output_ratio: float = input_dict["ndof_output_ratio"]

    ## Read tracked values from each strategy
    for ref_strat_name, ref_strat in refinement_strategies.items():
        ref_strat_dir_path: str = os.path.join(out_dir_path, ref_strat_name)

        tracked_values_file_name: str = "tracked_vals.json"
        tracked_values_file_path: str = os.path.join(ref_strat_dir_path,
                                                     tracked_values_file_name)

        ref_strat_err_dict: dict = {}
        ## If we already have the error history, then skip calculating it
        if not os.path.isfile(tracked_values_file_path):
            prev_ndof: int = 1

            subdir_paths: list = [subdir_path 
                                  for subdir_path in os.listdir(ref_strat_dir_path)
                                  if os.path.isdir(subdir_path)]
            trial_dir_paths: list = [os.path.join(ref_strat_dir_path, subdir_path) 
                                     for subdir_path in np.sort(np.array(subdir_paths, dtype = consts.INT)).astype(str)
                                     if os.path.isdir(os.path.join(ref_strat_dir_path, subdir_path))]
            for trial_dir_path in trial_dir_paths:
                ## Read the solution from file
                mesh_file_name: str = "mesh.json"
                mesh_file_path: str = os.path.join(trial_dir_path, mesh_file_name)

                uh_file_name: str = "uh.npy"
                uh_file_path: str = os.path.join(trial_dir_path, uh_file_name)

                uh: Projection = projection_from_file(mesh_file_path, uh_file_path)
                ndof: int = uh.mesh.get_ndof()

                ## Only calculate the analytic error every once in a while, or
                ## if it's the final one
                if ((float(ndof) / float(prev_ndof) >= ndof_output_ratio)
                    or (trial_dir_path == trial_dir_paths[-1])):
                    err_ind_file_name: str = "err_ind_anl.json"
                    err_ind_file_path: str = os.path.join(trial_dir_path, err_ind_file_name)

                    ## If we already have the analytic error, then just read it from file
                    if os.path.isfile(err_ind_file_path):
                        with open(err_ind_file_path, "r") as err_ind_file:
                            error: float = json.load(err_ind_file)["error"]
                            
                    else:
                        err_ind: Error_Indicator = Error_Indicator(uh, **ref_strat)
                        err_ind.error_analytic(u)
                        error: float = err_ind.error

                        ## Save the analytic error indicator to file
                        err_ind.to_file(err_ind_file_path, 
                                        write_projection = False, 
                                        write_mesh = False)

                    ref_strat_err_dict[ndof] = error

                    prev_ndof: int = ndof

            if comm_rank == consts.COMM_ROOT:
                with open(tracked_values_file_path, "w") as tracked_values_file:
                    json.dump(ref_strat_err_dict, tracked_values_file)

if __name__ == "__main__":
    main()