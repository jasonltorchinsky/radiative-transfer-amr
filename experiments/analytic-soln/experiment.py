import os, sys
src_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, os.pardir, "src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Standard Library Imports
import argparse
import gc
import json

from time import perf_counter

# Third-Party Library Imports
import numpy as np
import petsc4py
import psutil

from mpi4py   import MPI
from petsc4py import PETSc

# Local Library Imports
import dg.mesh as ji_mesh
import utils

import params
from gen_mesh_plot import gen_mesh_plot, gen_mesh_plot_p
from gen_uh_plot import gen_uh_plot
from gen_convergence_plot import gen_convergence_plot
from gen_err_plot import gen_err_plot
from get_soln import get_soln
from get_err import get_err
from ref_mesh import ref_mesh

def main(argv):
    ## Initialize mpi4py, petsc4py
    petsc4py.init()
    MPI_comm: MPI.Intracomm = MPI.COMM_WORLD # Communicator for passing data
    PETSc_comm: PETSc.Comm = PETSc.COMM_WORLD # Communicatior for matrices

    comm_rank: int = PETSc_comm.getRank()

    ## Read command-line input
    parser_desc: str = ( "Runs the numerical experiment for the hp-adaptive DG" +
                         " method for radiative transfer." )
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description = parser_desc)
    
    parser.add_argument("--o",
                        action = "store",
                        nargs = 1, 
                        type = str,
                        required = False,
                        default = "figs",
                        help = "Output directory path.")
    
    args = parser.parse_args()

    if (args.o != "figs"):
        out_dir_path: str = os.path.normpath(args.o[0])
    else:
        out_dir_path: str = args.o
    
    figs_dir_name: str = "figs"
    figs_dir: str = os.path.join(out_dir_path, figs_dir_name)
    os.makedirs(figs_dir, exist_ok = True)

    ## Unpack variables from params.py
    # Stopping conditions
    max_ndof: int   = params.stopping_conditions["max_ndof"]
    max_ntrial: int = params.stopping_conditions["max_ntrial"]
    min_err: int    = params.stopping_conditions["min_err"]
    max_mem: int    = params.stopping_conditions["max_mem"]

    # Refinement strategies
    ref_strats: list = params.ref_strats

    # Coefficients, forcing, and boundary conditions for this experiment
    kappa     = params.kappa
    sigma     = params.sigma
    Phi       = params.Phi
    f         = params.f
    bcs_dirac = params.bcs_dirac

    # Analtyic solution
    u = params.u
    u_intg_xy = params.u_intg_xy

    ## Set up RNG
    rng: np.random._generator.Generator = np.random.default_rng(params.seed)
    
    ## Output options
    do_plot_mesh: bool    = params.output_options["plot_mesh"]
    do_plot_mesh_p: bool  = params.output_options["plot_mesh_p"]
    do_plot_uh: bool      = params.output_options["plot_uh"]
    do_plot_err_ind: bool = params.output_options["plot_err_ind"]
    do_plot_errs: bool    = params.output_options["plot_errs"]
    
    ## Parameters to track across different refinement strategies
    ref_strat_names: list = []
    ref_strat_ndofs: dict = {}
    ref_strat_errs: dict  = {}
    ref_strat_nnz: dict   = {}
    
    ## Set up the timer
    perf_all_0 = perf_counter()
    msg: str = ( "Generating experiment figures...\n" )
    utils.print_msg(msg)
    
    for ref_strat in ref_strats:
        ## Create output directory for this refinement strategy
        ref_strat_name: str = ref_strat["short_name"]
        ref_strat_names += [ref_strat_name]
        ref_strat_dir: str = os.path.join(figs_dir, ref_strat_name)
        os.makedirs(ref_strat_dir, exist_ok = True)
        
        ## Initiate timers
        msg: str = ( "Starting combination {}...\n".format(ref_strat["full_name"]) )
        utils.print_msg(msg)
        
        perf_ref_strat_0: float = perf_counter()
        perf_setup_0: float = perf_counter()
        
        ## Generate the initial mesh
        if comm_rank == 0:
            mesh: ji_mesh.Mesh = ji_mesh.Mesh(Ls     = ref_strat["Ls"],
                                pbcs   = ref_strat["pbcs"],
                                ndofs  = ref_strat["ndofs"],
                                has_th = ref_strat["has_th"])
            
            for _ in range(0, ref_strat["nref_ang"]):
                mesh.ref_mesh(kind = "ang", form = "h")
                
            for _ in range(0, ref_strat["nref_spt"]):
                mesh.ref_mesh(kind = "spt", form = "h")
        else:
            mesh: ji_mesh.Mesh = None
        MPI_comm.Barrier()
        
        ## Parameters to track over each trial
        ndofs: list = []
        errs: list  = []
        nnz: list   = []
        refs: list  = []

        ## Parameters for stopping a refinement strategy
        if comm_rank == 0:
            ndof: int = mesh.get_ndof()
        else:
            ndof: int = None
        ndof: int = MPI_comm.bcast(ndof, root = 0)
        prev_ndof: int = ndof
        trial: int = 0
        err: float   = 1.
        mem_used: float = psutil.virtual_memory()[2]
        
        ## Parameters for how often to calculate error
        prev_err_ndof: int = 1
        
        ## Timer for setup
        perf_setup_f: float = perf_counter()
        perf_setup_diff: float = perf_setup_f - perf_setup_0
        msg: str = ( "Combination {} setup complete!\n".format(ref_strat["full_name"]) +
                12 * " " + "Time Elapsed: {:08.3f} [s]\n".format(perf_setup_diff)
               )
        utils.print_msg(msg)
        
        ## Main loop - Run the refinement strategy on this experiment
        while (((ndof < max_ndof)
                #and (trial <= max_ntrial)
                and (err > min_err)
                and (mem_used <= 95.))
               or (trial <= 1)):
            ## Update stopping criterion numbers
            mem_used: float = psutil.virtual_memory()[2]
            if comm_rank == 0:
                ndof: int = mesh.get_ndof()
            else:
                ndof: int = None
            ndof: int = MPI_comm.bcast(ndof, root = 0)
            
            ## Initiate timee for this trial
            perf_trial_0: float = perf_counter()
            msg: str = (
                "[Trial {}] Starting with: ".format(trial) +
                "{} of {} DoFs and\n".format(ndof, max_ndof) +
                37 * " " + "error {:.2E} of {:.2E}\n".format(err, min_err) +
                37 * " " + "RAM Memory % Used: {}\n".format(mem_used)
            )
            utils.print_msg(msg)
            
            ## Flag for calculating error
            do_calc_err = ((ndof / prev_err_ndof) >= 1.2 
                           or ((max_ndof / ndof) <= 1.1 
                               and ((ndof / prev_err_ndof) >= 1.1)))
            
            ## Set up output directories
            trial_dir: str = os.path.join(ref_strat_dir, "trial_{}".format(trial))
            os.makedirs(trial_dir, exist_ok = True)
            
            ## Plot mesh
            if comm_rank == 0:
                if (trial%10 == 0 or do_calc_err):
                    mesh_file = os.path.join(trial_dir, "mesh.json")
                    ji_mesh.write_mesh(mesh, mesh_file)
                    if do_plot_mesh:
                        gen_mesh_plot(mesh, trial, trial_dir, blocking = False)
                    if do_plot_mesh_p:
                        gen_mesh_plot_p(mesh, trial, trial_dir, blocking = False)
            MPI_comm.barrier()
            
            ## Set solver an preconditioner
            ksp_type: str = "lgmres"
            pc_type: str = "bjacobi"
                
            ## Calculate the numerical solution
            residual_file_name: str = "residuals_{}.png".format(trial)
            residual_file_path: str = os.path.join(trial_dir, residual_file_name)
            [uh_proj, info, mat_info] = get_soln(mesh, kappa, sigma, Phi, 
                                                 bcs_dirac, f, trial,
                                                 ksp_type = ksp_type,
                                                 pc_type = pc_type,
                                                 residual_file_path = residual_file_path)
            PETSc.garbage_cleanup()

            ## Plot the numerical solution
            if comm_rank == 0:
                if do_plot_uh and (trial%10 == 0 or do_calc_err):
                    gen_uh_plot(mesh, uh_proj, trial, trial_dir, blocking = False)
            MPI_comm.barrier()

            ## Calculate error
            if do_calc_err:
                err_kind: str = "anl"

                residual_file_name: str = "residuals_hr_{}.png".format(trial)
                residual_file_path: str = os.path.join(trial_dir, residual_file_name)
                    
                err = get_err(mesh, uh_proj, u, kappa, sigma, Phi,
                              bcs_dirac, f,
                              trial, trial_dir,
                              nref_ang  = ref_strat["nref_ang"],
                              nref_spt  = ref_strat["nref_spt"],
                              ref_kind  = ref_strat["ref_kind"],
                              spt_res_offset = ref_strat["spt_res_offset"],
                              ang_res_offset = ref_strat["ang_res_offset"],
                              key       = 1, # Leftover from when we depended on test number
                              err_kind  = err_kind,
                              ksp_type  = ksp_type,
                              pc_type   = pc_type,
                              file_path = figs_dir,
                              residual_file_path = residual_file_path)
                PETSc.garbage_cleanup()
                
                ndofs += [ndof]
                errs  += [err]
                nnz   += [int(mat_info["nz_used"])]
                prev_err_ndof = ndof
                
                if comm_rank == 0:
                    # Write error results to files as we go along
                    file_name = "errs.txt"
                    file_path = os.path.join(ref_strat_dir, file_name)
                    json.dump(errs, open(file_path, "w"))
                    
                    file_name = "ndofs.txt"
                    file_path = os.path.join(ref_strat_dir, file_name)
                    json.dump(ndofs, open(file_path, "w"))

                    file_name = "nnz.txt"
                    file_path = os.path.join(ref_strat_dir, file_name)
                    json.dump(nnz, open(file_path, "w"))
                
            # Refine the mesh, plot error indicators along the way
            [mesh, _, refs] = ref_mesh(ref_strat, mesh, uh_proj, comm_rank, 
                                       rng, do_plot_err_ind, trial, do_calc_err,
                                       trial_dir, ndof, info, ref_strat_dir,
                                       refs, u_intg_xy)
                    
            perf_trial_f    = perf_counter()
            perf_trial_diff = perf_trial_f - perf_trial_0
            if comm_rank == 0:
                ndof = mesh.get_ndof()
                ndof = MPI_comm.bcast(ndof, root = 0)
            else:
                ndof = None
                ndof = MPI_comm.bcast(ndof, root = 0)
            mem_used = psutil.virtual_memory()[2]
            msg = (
                "[Trial {}] Trial completed!\n".format(trial) +
                12 * " " + "Time Elapsed: {:08.3f} [s]\n".format(perf_trial_diff) +
                12 * " " + "Next trial: {} of {} DoFs and\n".format(ndof, max_ndof) +
                24 * " " + "error {:.2E} of {:.2E}\n".format(err, min_err) +
                24 * " " + "RAM Memory % Used: {}\n".format(mem_used)
            )
            utils.print_msg(msg)
            
            trial += 1
            
        if comm_rank == 0:
            # Write error results to files as we go along
            file_name = "errs.txt"
            file_path = os.path.join(ref_strat_dir, file_name)
            json.dump(errs, open(file_path, "w"))
            
            file_name = "ndofs.txt"
            file_path = os.path.join(ref_strat_dir, file_name)
            json.dump(ndofs, open(file_path, "w"))
            
            if do_plot_errs:
                gen_err_plot(ref_strat, ref_strat_name, ndofs, errs, ref_strat_dir)
                
                ref_strat_ndofs[ref_strat_name] = ndofs
                ref_strat_errs[ref_strat_name]  = errs
                ref_strat_nnz[ref_strat_name]   = mat_info["nz_used"]
                
            perf_ref_strat_f = perf_counter()
            perf_ref_strat_dt = perf_ref_strat_f - perf_ref_strat_0
            msg = (
                "Combination {} complete!\n".format(ref_strat["full_name"]) +
                12 * " " + "Time elapsed: {:08.3f} [s]\n".format(perf_ref_strat_dt)
            )
            utils.print_msg(msg, blocking = False)
            
        # Clear some variables to reduce memory usage
        del mesh, uh_proj
        gc.collect()
    
    if comm_rank == 0:
        if do_plot_errs:
            gen_convergence_plot(ref_strats, ref_strat_names, ref_strat_ndofs, 
                                 ref_strat_errs, figs_dir)
            
        perf_all_f = perf_counter()
        perf_all_dt = perf_all_f - perf_all_0
        msg = (
            "Experiment figures generated!\n" +
            12 * " " + "Time elapsed: {:08.3f} [s]\n".format(perf_all_dt)
        )
        utils.print_msg(msg, blocking = False)

if __name__ == "__main__":
    main(sys.argv[1:])