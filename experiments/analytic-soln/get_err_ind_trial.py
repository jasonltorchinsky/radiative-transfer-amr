import os, sys
src_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, os.pardir, "src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Standard Library Imports
import argparse
import json

# Third-Party Library Imports
import petsc4py
from mpi4py import MPI
from petsc4py import PETSc

# Local Library Imports
import consts
from amr.error_indicator import Error_Indicator
from dg.projection import Projection
from dg.projection import from_file as projection_from_file

# Relative Imports
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
    parser_desc: str = ( "Calculates all error indicators of interest given the " +
                         "low- and high-resolution numerical solutions.")
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
        trial_dir_path: str = os.path.normpath(args.o[0])
    else:
        trial_dir_path: str = args.o

    ## Read input - hardcoded file names
    input_file = open("input.json")
    input_dict: dict = json.load(input_file)
    input_file.close()

    hr_err_params: dict = input_dict["hr_err_params"]

    ## Read the low-resolution projection from file
    mesh_file_name: str = "mesh.json"
    mesh_file_path: str = os.path.join(trial_dir_path, mesh_file_name)
    
    uh_file_name: str = "uh.npy"
    uh_file_path: str = os.path.join(trial_dir_path, uh_file_name)

    if comm_rank == consts.COMM_ROOT:
        uh: Projection = projection_from_file(mesh_file_path, uh_file_path)
    else:
        uh: Projection = None
    uh: Projection = mpi_comm.bcast(uh, root = consts.COMM_ROOT)

    ## When calculating the error indicators here, it doesn't actually matter
    ## which refinement strategy we use, since we aren't using it to refine
    ## the mesh
    ref_strat: dict = sorted(refinement_strategies.items())[0][1]

    if comm_rank == consts.COMM_ROOT:
        ## Only calculate the error indicator if it isn't already on file

        ## Analytic error indicator
        err_ind_anl_file_name: str = "err_ind_anl.json"
        err_ind_anl_file_path: str = os.path.join(trial_dir_path, 
                                                  err_ind_anl_file_name)

        if not os.path.isfile(err_ind_anl_file_path):
            err_ind_anl: Error_Indicator = Error_Indicator(uh, **ref_strat)
            err_ind_anl.error_analytic(u)

            ## Save the analytic error indicator to file
            err_ind_anl.to_file(err_ind_anl_file_path, 
                                write_projection = False, 
                                write_mesh = False)

        ## Angular jump error indicator
        err_ind_jmp_file_name: str = "err_ind_jmp.json"
        err_ind_jmp_file_path: str = os.path.join(trial_dir_path, 
                                                  err_ind_jmp_file_name)
        if not os.path.isfile(err_ind_jmp_file_path):
            err_ind_jmp: Error_Indicator = Error_Indicator(uh, **ref_strat)
            err_ind_jmp.error_cell_jump()
            
            ## Save the cell jump error indicator to file
            err_ind_jmp.to_file(err_ind_jmp_file_path, 
                                write_projection = False, 
                                write_mesh = False)
    
    mpi_comm.barrier()

    ## High-resolution error indicator
    err_ind_hr_file_name: str = "err_ind_hr.json"
    err_ind_hr_file_path: str = os.path.join(trial_dir_path, 
                                             err_ind_hr_file_name)
    
    if not os.path.isfile(err_ind_hr_file_path):
        err_ind_hr: Error_Indicator = Error_Indicator(uh, **ref_strat)
        ## If we have the high-resolution solution on file, use that
        mesh_hr_file_name: str = "mesh_hr.json"
        mesh_hr_file_path: str = os.path.join(trial_dir_path, mesh_hr_file_name)

        uh_hr_file_name: str = "uh_hr.npy"
        uh_hr_file_path: str = os.path.join(trial_dir_path, uh_hr_file_name)
        if (os.path.isfile(mesh_hr_file_path) and os.path.isfile(uh_hr_file_path)):
            uh_hr: Projection = projection_from_file(mesh_hr_file_path, uh_hr_file_path)
        else:
            uh_hr: Projection = None
        [uh_hr, _, _] = err_ind_hr.error_high_resolution(problem, uh_hr = uh_hr,
                                                          **hr_err_params)

        if comm_rank == consts.COMM_ROOT:
            ## Save the high-resolution solution to file if it isn't already there
            if not os.path.isfile(uh_hr_file_name):
                uh_hr.to_file(uh_hr_file_path, wrte_mesh = True,
                              mesh_file_path = mesh_hr_file_path)

            ## Save the high-resolution error indicator to file
            err_ind_hr.to_file(err_ind_hr_file_path, 
                               write_projection = False, 
                               write_mesh = False)
            
    mpi_comm.barrier()


if __name__ == "__main__":
    main()