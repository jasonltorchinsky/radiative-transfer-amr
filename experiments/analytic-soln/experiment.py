import os, sys
src_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, os.pardir, "src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Standard Library Imports
import argparse
import copy
import json

# Third-Party Library Imports
import petsc4py
from mpi4py import MPI
from petsc4py import PETSc

# Local Library Imports
import consts
from dg.mesh import Mesh
from amr.error_indicator import Error_Indicator

# Relative Imports
from mesh import mesh
from generate_plots import generate_plots
from problem import problem, u
from refinement_strategies import refinement_strategies

def main():
    ## Initialize parallel communicators
    mpi_comm: MPI.Intracomm = MPI.COMM_WORLD
    
    if not PETSc.Sys.isInitialized():
        petsc4py.init(comm = mpi_comm)
    petsc_comm: PETSc.Comm = PETSc.COMM_WORLD
    comm_rank: int = petsc_comm.getRank()

    ## Read command-line input
    parser_desc: str = ( "Runs the numerical experiment for the hp-adaptive DG" +
                         " method for radiative transfer." )
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
    input_file = open("input.json")
    input_dict: dict = json.load(input_file)
    input_file.close()

    stopping_conditions: dict = input_dict["stopping_conditions"]
    ## If a stopping condition is 0 or less, it should be ignored
    for key, val in stopping_conditions.items():
        if val <= 0:
            stopping_conditions[key] = consts.INF

    ## Calculate, output, and plot the solution
    for ref_strat_name, ref_strat in refinement_strategies.items():
        ## Values to record for each refinement strategy
        tracked_vals: dict = {"ndof" : {},
                              "error": {}}

        if comm_rank == 0:
            ref_strat_dir_path: str = os.path.join(out_dir_path, ref_strat_name)
            os.makedirs(ref_strat_dir_path, exist_ok = True)

        l_mesh: Mesh = copy.deepcopy(mesh)

        ## Set values for output conditions
        prev_ndof: int = 1 # After we refine enough, generate plots

        ## Set values for stopping conditions
        trial: int = 0
        min_err: float = consts.INF
        keep_going: bool = ( (l_mesh.get_ndof() <= stopping_conditions["max_ndof"])
                            and (trial < stopping_conditions["max_ntrial"])
                            and (min_err >= stopping_conditions["min_err"]) )
        while (keep_going):
            if comm_rank == 0:
                trial_dir_path: str = os.path.join(ref_strat_dir_path,
                                                   str(trial))
                os.makedirs(trial_dir_path, exist_ok = True)

            [uh, _, _] = problem.solve(l_mesh)

            if comm_rank == 0:
                ## Save the numeric solution to file
                proj_file_name: str = "uh.npy"
                proj_file_path: str = os.path.join(trial_dir_path, proj_file_name)

                mesh_file_name: str = "mesh.json"
                mesh_file_path: str = os.path.join(trial_dir_path, mesh_file_name)

                uh.to_file(proj_file_path, wrte_mesh = True,
                           mesh_file_path = mesh_file_path)

                ## Calculate the error indicator
                err_ind: Error_Indicator = Error_Indicator(uh, **ref_strat)
                err_ind.error_analytic(u)

                ## Updates values to record
                tracked_vals["ndof"][trial] = l_mesh.get_ndof
                tracked_vals["error"][trial] = err_ind.error

            keep_going: bool = ( (l_mesh.get_ndof() <= stopping_conditions["max_ndof"])
                                  and (trial <= stopping_conditions["max_ntrial"])
                                  and (min_err >= stopping_conditions["min_err"]) )
            
            ## if keep_going: refine the mesh, occassionally generate plots
            ## and update tracked values
            ## else: generate plots and update tracked values
            if keep_going:
                if comm_rank == 0:
                    if (float(l_mesh.get_ndof()) / float(prev_ndof) >= 1.25):
                        generate_plots(uh, err_ind, trial_dir_path)

                        file_name: str = "tracked_vals.json"
                        file_path = os.path.join(ref_strat_dir_path, file_name)
                        with open(file_path, "w") as file:
                            json.dump(tracked_vals, file)
                    l_mesh: Mesh = err_ind.ref_by_ind()
                l_mesh: Mesh = mpi_comm.bcast(l_mesh, root = 0)

                ## Update values for stopping conditions
                trial += 1
                if comm_rank == 0:
                    min_err: float = min(min_err, err_ind.error)
                min_err: float = mpi_comm.bcast(min_err, root = 0)
            else:
                if comm_rank == 0:
                    generate_plots(uh, err_ind, trial_dir_path)

                    file_name: str = "tracked_vals.json"
                    file_path = os.path.join(ref_strat_dir_path, file_name)
                    with open(file_path, "w") as file:
                        json.dump(tracked_vals, file)

if __name__ == "__main__":
    main()