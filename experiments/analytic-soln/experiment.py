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

            [uh, _, _] = problem.solve(l_mesh)
            PETSc.garbage_cleanup(petsc_comm)

            if comm_rank == consts.COMM_ROOT:
                ## Save the numeric solution to file
                proj_file_name: str = "uh.npy"
                proj_file_path: str = os.path.join(trial_dir_path, proj_file_name)

                mesh_file_name: str = "mesh.json"
                mesh_file_path: str = os.path.join(trial_dir_path, mesh_file_name)

                uh.to_file(proj_file_path, wrte_mesh = True,
                           mesh_file_path = mesh_file_path)
                
                ## Calculate the cell jump error indicator
                err_ind_jmp: Error_Indicator = Error_Indicator(uh, **ref_strat)
                err_ind_jmp.error_cell_jump()

                ## Save the cell jump error indicator to file
                err_ind_jmp_file_name: str = "err_ind_jmp.json"
                err_ind_jmp_file_path: str = os.path.join(trial_dir_path, 
                                                          err_ind_jmp_file_name)
                err_ind_jmp.to_file(err_ind_jmp_file_path, 
                                    write_projection = False, 
                                    write_mesh = False)

                ## Calculate the analytic error indicator
                err_ind_anl: Error_Indicator = Error_Indicator(uh, **ref_strat)
                err_ind_anl.error_analytic(u)

                ## Save the analytic error indicator to file
                err_ind_anl_file_name: str = "err_ind_anl.json"
                err_ind_anl_file_path: str = os.path.join(trial_dir_path, 
                                                          err_ind_anl_file_name)
                err_ind_anl.to_file(err_ind_anl_file_path, 
                                    write_projection = False, 
                                    write_mesh = False)

            ## Calculate the high-resolution error indicator
            err_ind_hr: Error_Indicator = Error_Indicator(uh, **ref_strat)
            [uh_hr, _, _] = err_ind_hr.error_high_resolution(problem, **hr_err_params)

            if comm_rank == consts.COMM_ROOT:
                ## Save the high-resolution solution to file
                proj_file_name: str = "uh_hr.npy"
                proj_file_path: str = os.path.join(trial_dir_path, proj_file_name)

                mesh_file_name: str = "mesh_hr.json"
                mesh_file_path: str = os.path.join(trial_dir_path, mesh_file_name)

                uh_hr.to_file(proj_file_path, wrte_mesh = True,
                              mesh_file_path = mesh_file_path)
                
                ## Save the high-resolution error indicator to file
                err_ind_hr_file_name: str = "err_ind_hr.json"
                err_ind_hr_file_path: str = os.path.join(trial_dir_path, 
                                                          err_ind_hr_file_name)
                err_ind_hr.to_file(err_ind_hr_file_path, 
                                   write_projection = False, 
                                   write_mesh = False)

            ## We refine the mesh here because if it gets too big, then we'll
            ## want to stop. However, this doesn *not* update the mesh in the
            ## uh and err_ind objects
            if comm_rank == consts.COMM_ROOT:
                l_mesh: Mesh = err_ind_hr.ref_by_ind()
            l_mesh: Mesh = mpi_comm.bcast(l_mesh, root = consts.COMM_ROOT)

            ## Update values for stopping conditions - these reflect what 
            ## the next trial *will* be
            trial += 1
            if comm_rank == consts.COMM_ROOT:
                min_err: float = min(min_err, err_ind_anl.error)
            min_err: float = mpi_comm.bcast(min_err, root = consts.COMM_ROOT)
            
            keep_going: bool = ( (l_mesh.get_ndof() <= stopping_conditions["max_ndof"])
                                  and (trial <= stopping_conditions["max_ntrial"])
                                  and (min_err >= stopping_conditions["min_err"]) )
                

if __name__ == "__main__":
    main()