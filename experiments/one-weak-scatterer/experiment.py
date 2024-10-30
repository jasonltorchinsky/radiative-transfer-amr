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
from problem import problem
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
    with open("input.json", "r") as input_file:
        input_dict: dict = json.load(input_file)

    ndof_output_ratio: float = input_dict["ndof_output_ratio"]
    stopping_conditions: dict = input_dict["stopping_conditions"]
    solver_params: dict = input_dict["solver_params"]
    hr_err_params: dict = input_dict["hr_err_params"]
    
    ## If a stopping condition is 0 or less, it should be ignored
    for key, val in stopping_conditions.items():
        if val <= 0:
            stopping_conditions[key] = consts.INF

    ## Calculate, output, and plot the solution
    for ref_strat_name, ref_strat in refinement_strategies.items():
        if comm_rank == consts.COMM_ROOT:
            ref_strat_dir_path: str = os.path.join(out_dir_path, ref_strat_name)
            os.makedirs(ref_strat_dir_path, exist_ok = True)

            ## Set up the log for this refinement strategy
            ref_strat_log: dict = {}
            ref_strat_log_file_name: str = "ref_strat_log.json"
            ref_strat_log_file_path: str = os.path.join(ref_strat_dir_path,
                                                        ref_strat_log_file_name)

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
            if comm_rank == consts.COMM_ROOT:
                trial_dir_path: str = os.path.join(ref_strat_dir_path,
                                                   str(trial))
                os.makedirs(trial_dir_path, exist_ok = True)

                ref_strat_log[trial] = {}

            [uh, convergence_info, matrix_info] = problem.solve(l_mesh, **solver_params)
            PETSc.garbage_cleanup(petsc_comm)

            if comm_rank == consts.COMM_ROOT:
                ## Save the linear solve information to the refinement strategy log
                ref_strat_log[trial]["lr_converged_reason"] = convergence_info["converged_reason"]
                ref_strat_log[trial]["lr_res_best"] = convergence_info["res_best"]

                ## Save the linear solve and matrix information to file
                convergence_info_file_name: str = "convergence_info.json"
                convergence_info_file_path: str = os.path.join(trial_dir_path,
                                                               convergence_info_file_name)
                with open(convergence_info_file_path, "w") as convergence_info_file:
                    json.dump(convergence_info, convergence_info_file)

                matrix_info_file_name: str = "matrix_info.json"
                matrix_info_file_path: str = os.path.join(trial_dir_path,
                                                          matrix_info_file_name)
                with open(matrix_info_file_path, "w") as matrix_info_file:
                    json.dump(matrix_info, matrix_info_file)
                
                ## Save the numeric solution to file
                proj_file_name: str = "uh.npy"
                proj_file_path: str = os.path.join(trial_dir_path, proj_file_name)

                mesh_file_name: str = "mesh.json"
                mesh_file_path: str = os.path.join(trial_dir_path, mesh_file_name)

                uh.to_file(proj_file_path, write_mesh = True,
                           mesh_file_path = mesh_file_path)
                
                ## Calculate the angular jump error indicator
                err_ind_ang_jmp: Error_Indicator = Error_Indicator(uh, **ref_strat)
                err_ind_ang_jmp.error_angular_jump()

                ## Save the angular jump error indicator to file
                err_ind_ang_jmp_file_name: str = "err_ind_ang_jmp.json"
                err_ind_ang_jmp_file_path: str = os.path.join(trial_dir_path, 
                                                              err_ind_ang_jmp_file_name)
                err_ind_ang_jmp.to_file(err_ind_ang_jmp_file_path, 
                                        write_projection = False, 
                                        write_mesh = False)

            ## Only calculate the analytic and high-resolution error every
            ## once in a while
            if (l_mesh.get_ndof() / prev_ndof >= ndof_output_ratio):
                ## Calculate the high-resolution error indicator
                err_ind_hr: Error_Indicator = Error_Indicator(uh, **ref_strat)
                [uh_hr, convergence_info_hr, matrix_info_hr] = \
                    err_ind_hr.error_high_resolution(problem, 
                                                     **solver_params,
                                                     **hr_err_params)
                PETSc.garbage_cleanup(petsc_comm)
    
                if comm_rank == consts.COMM_ROOT:
                    ## Save the linear solve information to the refinement strategy log
                    ref_strat_log[trial]["hr_converged_reason"] = convergence_info_hr["converged_reason"]
                    ref_strat_log[trial]["hr_res_best"] = convergence_info_hr["res_best"]
                    
                    ## Save the linear solve and matrix information to file
                    convergence_info_hr_file_name: str = "convergence_info_hr.json"
                    convergence_info_hr_file_path: str = os.path.join(trial_dir_path,
                                                                      convergence_info_hr_file_name)
                    with open(convergence_info_hr_file_path, "w") as convergence_info_hr_file:
                        json.dump(convergence_info_hr, convergence_info_hr_file)

                    matrix_info_hr_file_name: str = "matrix_info_hr.json"
                    matrix_info_hr_file_path: str = os.path.join(trial_dir_path,
                                                              matrix_info_hr_file_name)
                    with open(matrix_info_hr_file_path, "w") as matrix_info_hr_file:
                        json.dump(matrix_info_hr, matrix_info_hr_file)

                    ## Save the high-resolution solution to file
                    proj_file_name: str = "uh_hr.npy"
                    proj_file_path: str = os.path.join(trial_dir_path, proj_file_name)
    
                    mesh_file_name: str = "mesh_hr.json"
                    mesh_file_path: str = os.path.join(trial_dir_path, mesh_file_name)
    
                    uh_hr.to_file(proj_file_path, write_mesh = True,
                                  mesh_file_path = mesh_file_path)
                    
                    ## Save the high-resolution error indicator to file
                    err_ind_hr_file_name: str = "err_ind_hr.json"
                    err_ind_hr_file_path: str = os.path.join(trial_dir_path, 
                                                             err_ind_hr_file_name)
                    err_ind_hr.to_file(err_ind_hr_file_path, 
                                       write_projection = False, 
                                       write_mesh = False)
                    
                prev_ndof: int = l_mesh.get_ndof()

            ## We refine the mesh here because if it gets too big, then we'll
            ## want to stop. However, this doesn *not* update the mesh in the
            ## uh and err_ind objects
            if comm_rank == consts.COMM_ROOT:
                l_mesh: Mesh = err_ind_ang_jmp.ref_by_ind()
            l_mesh: Mesh = mpi_comm.bcast(l_mesh, root = consts.COMM_ROOT)

            ## Update values for stopping conditions - these reflect what 
            ## the next trial *will* be
            trial += 1
            if comm_rank == consts.COMM_ROOT:
                min_err: float = min(min_err, err_ind_hr.error)
            min_err: float = mpi_comm.bcast(min_err, root = consts.COMM_ROOT)
            
            keep_going: bool = ( (l_mesh.get_ndof() <= stopping_conditions["max_ndof"])
                                  and (trial <= stopping_conditions["max_ntrial"])
                                  and (min_err >= stopping_conditions["min_err"]) )
            
        if comm_rank == consts.COMM_ROOT:
            with open(ref_strat_log_file_path, "w") as ref_strat_log_file:
                json.dump(ref_strat_log, ref_strat_log_file)
                

if __name__ == "__main__":
    main()