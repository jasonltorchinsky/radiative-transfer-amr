"""
Script to generate the figures included in the publication
"""

# Standard Library Imports
import argparse
import gc
import json
import os
import sys

from   time import perf_counter

# Third-Party Library Imports
import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
import petsc4py
import psutil

from   mpi4py          import MPI
from   petsc4py        import PETSc
from   scipy.integrate import quad, dblquad

# Local Library Imports
sys.path.append("../src")
import amr
import amr.utils
import dg.matrix     as mat
import dg.mesh       as ji_mesh
import dg.mesh.utils
import dg.projection as proj
import dg.quadrature as qd
import rt
import utils

import test_temp as test

def main():
    petsc4py.init()

    # Get local copy of variables from test.py - which is a temporary copy of a
    # test_n.py for some n
    max_ndof   = test.max_ndof
    max_ntrial = test.max_ntrial
    min_err    = test.min_err
    max_mem    = test.max_mem
    [Lx, Ly]   = [test.Lx, test.Ly]
    combos     = test.combos
    kappa      = test.kappa
    sigma      = test.sigma
    Phi        = test.Phi
    f          = test.f
    bcs_dirac  = test.bcs_dirac
    u          = test.u
    
    # MPI COMM for communicating data
    MPI_comm = MPI.COMM_WORLD
    
    #PETSc COMM for parallel matrix construction, solves
    comm = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()
    
    parser_desc = "Determine which tests to run and where to put output."
    parser = argparse.ArgumentParser(description = parser_desc)
    
    parser.add_argument("--dir", nargs = 1, default = "figs",
                        required = False, help = "Subdirectory to store output")
    help_str = "Test Case Number - See Paper for Details"
    parser.add_argument("--test_num", nargs = 1, default = [1],
                        type = int, required = False,
                        help = help_str)
    
    args = parser.parse_args()
    dir_name = args.dir
    test_num = args.test_num[0]
    
    figs_dir_name = "test_{}_figs".format(test_num)
    figs_dir = os.path.join(dir_name, figs_dir_name)
    os.makedirs(figs_dir, exist_ok = True)

    # Set up RNG
    rng = np.random.default_rng()
    
    # Output options
    do_plot_mesh        = False
    do_plot_mesh_p      = True
    do_plot_uh          = True
    do_plot_err_ind     = True
    do_plot_errs        = True
    
    combo_names = []
    combo_ndofs = {}
    combo_errs  = {}
    combo_nnz   = {}
    
    perf_all_0 = perf_counter()
    msg = ( "Generating test {} figures...\n".format(test_num) )
    utils.print_msg(msg)

    # Plot extinction coefficient, scattering coefficient, and scattering phase function
    if comm_rank == 0:
        kappa_file_name = "kappa_{}.png".format(test_num)
        sigma_file_name = "sigma_{}.png".format(test_num)
        Phi_file_name   = "Phi_{}.png".format(test_num)
        gen_kappa_sigma_plots([Lx, Ly], kappa, sigma, figs_dir,
                              [kappa_file_name, sigma_file_name])
        gen_Phi_plot(Phi, figs_dir, Phi_file_name)
        if test_num == 1:
            gen_u_plot([Lx, Ly], u, figs_dir)
    MPI_comm.Barrier()
    
    for combo in combos:
        combo_name = combo["short_name"]
        combo_names += [combo_name]
        combo_dir = os.path.join(figs_dir, combo_name)
        os.makedirs(combo_dir, exist_ok = True)
        
        msg = ( "Starting combination {}...\n".format(combo["full_name"]) )
        utils.print_msg(msg)
        
        perf_combo_0 = perf_counter()
        perf_setup_0 = perf_counter()
        
        # Get the base mesh
        if comm_rank == 0:
            mesh = ji_mesh.Mesh(Ls     = combo["Ls"],
                                pbcs   = combo["pbcs"],
                                ndofs  = combo["ndofs"],
                                has_th = combo["has_th"])
            
            for _ in range(0, combo["nref_ang"]):
                mesh.ref_mesh(kind = "ang", form = "h")
                
            for _ in range(0, combo["nref_spt"]):
                mesh.ref_mesh(kind = "spt", form = "h")
        else:
            mesh = None
        MPI_comm.Barrier()
        
        # Solve the problem over several trials
        ndofs = []
        errs  = []
        nnz   = []
        refs  = []

        # Parameters for stopping an experiment
        if comm_rank == 0:
            ndof = mesh.get_ndof()
        else:
            ndof = None
        ndof = MPI_comm.bcast(ndof, root = 0)
        prev_ndof = ndof
        trial = 0
        err   = 1.
        mem_used = psutil.virtual_memory()[2]
        # Parameters for how often to calculate error
        prev_err_ndof = 1
        
        perf_setup_f = perf_counter()
        perf_setup_diff = perf_setup_f - perf_setup_0
        msg = ( "Combination {} setup complete!\n".format(combo["full_name"]) +
                12 * " " + "Time Elapsed: {:08.3f} [s]\n".format(perf_setup_diff)
               )
        utils.print_msg(msg)
        
        while (((ndof < max_ndof)
                #and (trial <= max_ntrial)
                and (err > min_err)
                and (mem_used <= 95.))
               or (trial <= 1)):
            mem_used = psutil.virtual_memory()[2]
            if comm_rank == 0:
                ndof = mesh.get_ndof()
            else:
                ndof = None
            ndof = MPI_comm.bcast(ndof, root = 0)
            
            perf_trial_0 = perf_counter()
            msg = (
                "[Trial {}] Starting with: ".format(trial) +
                "{} of {} DoFs and\n".format(ndof, max_ndof) +
                37 * " " + "error {:.2E} of {:.2E}\n".format(err, min_err) +
                37 * " " + "RAM Memory % Used: {}\n".format(mem_used)
            )
            utils.print_msg(msg)
            
            do_calc_err = ((ndof / prev_err_ndof) >= 1.2 or ((max_ndof / ndof) <= 1.1 and ((ndof / prev_err_ndof) >= 1.1)))
            
            # Set up output directories
            trial_dir = os.path.join(combo_dir, "trial_{}".format(trial))
            os.makedirs(trial_dir, exist_ok = True)
            
            # Plot mesh
            if comm_rank == 0:
                if (trial%10 == 0 or do_calc_err):
                    mesh_file = os.path.join(trial_dir, "mesh.json")
                    ji_mesh.write_mesh(mesh, mesh_file)
                    if do_plot_mesh:
                        gen_mesh_plot(mesh, trial, trial_dir, blocking = False)
                    if do_plot_mesh_p:
                        gen_mesh_plot_p(mesh, trial, trial_dir, blocking = False)
            MPI_comm.barrier()
            
            # Get and plot numerical solution
            if test_num == 1:
                ksp_type = "lgmres"
                pc_type = "bjacobi"
            elif test_num == 2:
                ksp_type = "fgmres"
                pc_type = "kaczmarz"
            elif test_num == 3:
                ksp_type = "cgs"
                pc_type = "kaczmarz"
            elif test_num == 4:
                ksp_type = "fgmres"
                pc_type = "kaczmarz"
            else:
                ksp_type = "fgmres"
                pc_type = "kaczmarz"
                
            # If iterative solve fails, we refine the mesh
            residual_file_name = "residuals_{}.png".format(trial)
            residual_file_path = os.path.join(trial_dir, residual_file_name)
            [uh_proj, info, mat_info] = get_soln(mesh, kappa, sigma, Phi, bcs_dirac, f,
                                       trial,
                                       ksp_type = ksp_type,
                                       pc_type = pc_type,
                                       residual_file_path = residual_file_path)
            PETSc.garbage_cleanup()
            # Plot mesh after the solve in case it gets refined
            if comm_rank == 0:
                if do_plot_uh and (trial%10 == 0 or do_calc_err):
                    gen_uh_plot(mesh, uh_proj, trial, trial_dir, blocking = False)
            MPI_comm.barrier()

            # Get error and save it to file every so often
            if do_calc_err:
                if test_num == 1:
                    err_kind = "anl"
                else:
                    err_kind = "hr"

                residual_file_name = "residuals_hr_{}.png".format(trial)
                residual_file_path = os.path.join(trial_dir, residual_file_name)
                    
                err = get_err(mesh, uh_proj, u, kappa, sigma, Phi,
                              bcs_dirac, f,
                              trial, trial_dir,
                              nref_ang  = combo["nref_ang"],
                              nref_spt  = combo["nref_spt"],
                              ref_kind  = combo["ref_kind"],
                              spt_res_offset = combo["spt_res_offset"],
                              ang_res_offset = combo["ang_res_offset"],
                              key       = test_num,
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
                    file_path = os.path.join(combo_dir, file_name)
                    json.dump(errs, open(file_path, "w"))
                    
                    file_name = "ndofs.txt"
                    file_path = os.path.join(combo_dir, file_name)
                    json.dump(ndofs, open(file_path, "w"))

                    file_name = "nnz.txt"
                    file_path = os.path.join(combo_dir, file_name)
                    json.dump(nnz, open(file_path, "w"))
                
            # Refine the mesh, plot error indicators along the way
            if   combo["short_name"] == "h-uni-ang":
                if comm_rank == 0:
                    mesh.ref_mesh(kind = "ang", form = "h")
            elif combo["short_name"] == "p-uni-ang":
                if comm_rank == 0:
                    for _ in range(0, 3):
                        mesh.ref_mesh(kind = "ang", form = "p")
            elif combo["short_name"] == "hp-uni-ang":
                if comm_rank == 0:
                    for _ in range(0, 2):
                        mesh.ref_mesh(kind = "ang", form = "p")
                    mesh.ref_mesh(kind = "ang", form = "h")
            elif combo["short_name"] == "h-uni-spt":
                if comm_rank == 0:
                    mesh.ref_mesh(kind = "spt", form = "h")
            elif combo["short_name"] == "p-uni-spt":
                if comm_rank == 0:
                    for _ in range(0, 3):
                        mesh.ref_mesh(kind = "spt", form = "p")
            elif combo["short_name"] == "hp-uni-spt":
                if comm_rank == 0:
                    for _ in range(0, 2):
                        mesh.ref_mesh(kind = "spt", form = "p")
                    mesh.ref_mesh(kind = "spt", form = "h")
            elif combo["short_name"] == "h-uni-all":
                if comm_rank == 0:
                    mesh.ref_mesh(kind = "all", form = "h")
            elif combo["short_name"] == "p-uni-all":
                if comm_rank == 0:
                    for _ in range(0, 3):
                        mesh.ref_mesh(kind = "all", form = "p")
            elif combo["short_name"] == "hp-uni-all":
                if comm_rank == 0:
                    for _ in range(0, 2):
                        mesh.ref_mesh(kind = "all", form = "p")
                    mesh.ref_mesh(kind = "all", form = "h")
            elif ((combo["short_name"] == "h-amr-ang")
                  or (combo["short_name"] == "p-amr-ang")
                  or (combo["short_name"] == "hp-amr-ang")):
                if comm_rank == 0:
                    # Have two choices: Refine based on nneg or jump, and if
                    # also refining in space
                    if combo["ref_kind"] == "all":
                        # Calculate nneg errors
                        kwargs_ang_nneg  = combo["kwargs_ang_nneg"]
                        cell_ref_tol     = kwargs_ang_nneg["cell_ref_tol"]
                        nneg_ang_err_ind = amr.nneg_err_ang(mesh, uh_proj,
                                                            **kwargs_ang_nneg)
                        
                        # Spatial refinement strategy is uniform p-refinement
                        col_ref_tol = 1. # Positive refinement tolerance for
                        # nneg error means we"ll refine everything
                        kwargs_spt_nneg = {"ref_col"      : True,
                                           "col_ref_form" : "p",
                                           "col_ref_kind" : "spt",
                                           "col_ref_tol"  : col_ref_tol,
                                           "ref_cell"      : False,
                                           "cell_ref_form" : None,
                                           "cell_ref_kind" : None,
                                           "cell_ref_tol"  : None}
                        nneg_spt_err_ind = amr.nneg_err_spt(mesh, uh_proj,
                                                            **kwargs_spt_nneg)
                        
                        # If nneg error exceeds tolerance, we refine whichever
                        # has larger negative values in columns to be refined.
                        if ((nneg_ang_err_ind.cell_max_err < cell_ref_tol)
                            or (nneg_spt_err_ind.col_max_err < cell_ref_tol)):
                            # Use cell_ref_tol here because otherwise this
                            # would always trigger.
                            err_ind_ang = nneg_ang_err_ind
                            err_ind_spt = nneg_spt_err_ind

                            avg_cell_ref_err = nneg_ang_err_ind.avg_cell_ref_err
                            avg_col_ref_err  = nneg_spt_err_ind.avg_col_ref_err
                            if np.abs(avg_cell_ref_err) > np.abs(avg_col_ref_err):
                                err_ind = nneg_ang_err_ind
                                ref_str = "ang_nneg"
                            else:
                                err_ind = nneg_spt_err_ind
                                ref_str = "spt_nneg"
                                
                        else:
                            # If not using nneg err, then we use the jump error
                            kwargs_ang_jmp  = combo["kwargs_ang_jmp"]
                            jmp_ang_err_ind = amr.cell_jump_err(mesh, uh_proj,
                                                                **kwargs_ang_jmp)
                            col_ref_tol = -1. # Want uniform refinement
                            kwargs_spt_jmp = {"ref_col"      : True,
                                              "col_ref_form" : "p",
                                              "col_ref_kind" : "spt",
                                              "col_ref_tol"  : col_ref_tol,
                                              "ref_cell"      : False,
                                              "cell_ref_form" : None,
                                              "cell_ref_kind" : None,
                                              "cell_ref_tol"  : None}
                            jmp_spt_err_ind = amr.col_jump_err(mesh, uh_proj,
                                                               **kwargs_spt_jmp)
                            msg = (
                                "Comparison of average refinement error:\n" +
                                12 * " " + "Angular {:.4E} ".format(jmp_ang_err_ind.avg_cell_ref_err) +
                                "vs. Spatial {:.4E}\n".format(jmp_spt_err_ind.avg_col_ref_err)
                            )
                            utils.print_msg(msg, blocking = False)
                            err_ind_ang = jmp_ang_err_ind
                            err_ind_spt = jmp_spt_err_ind
                            
                            avg_cell_ref_err = jmp_ang_err_ind.avg_cell_ref_err
                            avg_col_ref_err  = jmp_spt_err_ind.avg_col_ref_err
                            
                            # Randomly choose which, but probability depends on steering criterion
                            p_ang = avg_cell_ref_err / (avg_cell_ref_err + avg_col_ref_err)
                            ref_strs = ["ang_jmp", "spt_jmp"]
                            ref_str = rng.choice(ref_strs, size = 1, p = (p_ang, 1 - p_ang))[0]
                            #if p_ang >= 0.5:
                            #    ref_str = ref_strs[0]
                            #else:
                            #    ref_str = ref_strs[1]
                            if ref_str == "ang_jmp":
                                err_ind = jmp_ang_err_ind
                            else: # ref_str == "spt_jmp"
                                err_ind = jmp_spt_err_ind
                        if do_plot_err_ind and (trial%10 == 0 or do_calc_err):
                            gen_err_ind_plot(mesh, err_ind_ang, trial, trial_dir, "err_ind_ang.png")
                            gen_err_ind_plot(mesh, err_ind_spt, trial, trial_dir, "err_ind_spt.png")
                        avg_cell_ref_err_str = "{:.4E}".format(jmp_ang_err_ind.avg_cell_ref_err)
                        avg_col_ref_err_str = "{:.4E}".format(jmp_spt_err_ind.avg_col_ref_err)
                        refs += [[ndof, ref_str, avg_cell_ref_err_str, avg_col_ref_err_str, info]]
                    else: # Just refining in angle
                        # Calculate nneg error
                        kwargs_ang_nneg  = combo["kwargs_ang_nneg"]
                        cell_ref_tol     = kwargs_ang_nneg["cell_ref_tol"]
                        nneg_ang_err_ind = amr.nneg_err_ang(mesh, uh_proj,
                                                            **kwargs_ang_nneg)
                        
                        # If nneg error exceeds tolerance, we refine whichever
                        # has larger negative values in columns to be refined.
                        if (nneg_ang_err_ind.cell_max_err < cell_ref_tol):
                            err_ind = nneg_ang_err_ind
                            ref_str = "ang_nneg"
                            
                        else:
                            # If not using nneg err, then we use the jump error
                            kwargs_ang_jmp  = combo["kwargs_ang_jmp"]
                            jmp_ang_err_ind = amr.cell_jump_err(mesh, uh_proj,
                                                                **kwargs_ang_jmp)
                            
                            err_ind = jmp_ang_err_ind
                            ref_str = "ang_jmp"
                        if do_plot_err_ind and (trial%10 == 0 or do_calc_err):
                            gen_err_ind_plot(mesh, err_ind, trial, trial_dir, "err_ind_ang.png")
                        refs += [[ndof, ref_str, info]]
                    
                    file_name = "refs.txt"
                    file_path = os.path.join(combo_dir, file_name)
                    json.dump(refs, open(file_path, "w"))

                    msg = (
                        "[Trial {}] Refining cause: {}\n".format(trial, ref_str)
                        )
                    utils.print_msg(msg, blocking = False)
                    mesh = amr.ref_by_ind(mesh, err_ind)
                    
            elif ((combo["short_name"] == "h-amr-spt")
                  or (combo["short_name"] == "p-amr-spt")
                  or (combo["short_name"] == "hp-amr-spt")):
                if comm_rank == 0:
                    # Have two choices: Refine based on nneg or jump, and if
                    # also refining in space
                    if combo["ref_kind"] == "all":
                        # Calculate nneg errors
                        kwargs_spt_nneg  = combo["kwargs_spt_nneg"]
                        col_ref_tol      = kwargs_spt_nneg["col_ref_tol"]
                        nneg_spt_err_ind = amr.nneg_err_spt(mesh, uh_proj,
                                                            **kwargs_spt_nneg)
                        
                        # Angular refinement strategy is uniform p-refinement
                        cell_ref_tol = 1. # Positive refinement tolerance for
                        # nneg error means we"ll refine everything
                        kwargs_ang_nneg = {"ref_col"      : False,
                                           "col_ref_form" : None,
                                           "col_ref_kind" : None,
                                           "col_ref_tol"  : None,
                                           "ref_cell"      : True,
                                           "cell_ref_form" : "p",
                                           "cell_ref_kind" : "ang",
                                           "cell_ref_tol"  : cell_ref_tol}
                        nneg_ang_err_ind = amr.nneg_err_ang(mesh, uh_proj,
                                                            **kwargs_ang_nneg)
                        
                        # If nneg error exceeds tolerance, we refine whichever
                        # has larger negative values in columns to be refined.
                        if ((nneg_spt_err_ind.col_max_err < col_ref_tol)
                            or (nneg_ang_err_ind.cell_max_err < col_ref_tol)):
                            # Use col_ref_tol here because otherwise this
                            # would always trigger.
                            err_ind_ang = nneg_ang_err_ind
                            err_ind_spt = nneg_spt_err_ind

                            avg_cell_ref_err = nneg_ang_err_ind.avg_cell_ref_err
                            avg_col_ref_err  = nneg_spt_err_ind.avg_col_ref_err
                            if np.abs(avg_cell_ref_err) > np.abs(avg_col_ref_err):
                                err_ind = nneg_ang_err_ind
                                ref_str = "ang_nneg"
                            else:
                                err_ind = nneg_spt_err_ind
                                ref_str = "spt_nneg"
                                
                        else:
                            # If not using nneg err, then we use the jump error
                            kwargs_spt_jmp  = combo["kwargs_spt_jmp"]
                            jmp_spt_err_ind = amr.col_jump_err(mesh, uh_proj,
                                                               **kwargs_spt_jmp)
                            cell_ref_tol = -1. # Want uniform refinement
                            kwargs_ang_jmp = {"ref_col"      : False,
                                              "col_ref_form" : None,
                                              "col_ref_kind" : None,
                                              "col_ref_tol"  : None,
                                              "ref_cell"      : True,
                                              "cell_ref_form" : "p",
                                              "cell_ref_kind" : "ang",
                                              "cell_ref_tol"  : cell_ref_tol}
                            jmp_ang_err_ind = amr.cell_jump_err(mesh, uh_proj,
                                                               **kwargs_ang_jmp)
                            msg = (
                                "Comparison of average refinement error:\n" +
                                12 * " " + "Angular {:.4E} ".format(jmp_ang_err_ind.avg_cell_ref_err) +
                                "vs. Spatial {:.4E}\n".format(jmp_spt_err_ind.avg_col_ref_err)
                            )
                            utils.print_msg(msg, blocking = False)
                            err_ind_ang = jmp_ang_err_ind
                            err_ind_spt = jmp_spt_err_ind

                            avg_cell_ref_err = jmp_ang_err_ind.avg_cell_ref_err
                            avg_col_ref_err  = jmp_spt_err_ind.avg_col_ref_err
                            
                            # Randomly choose which, but probability depends on steering criterion
                            p_ang = avg_cell_ref_err / (avg_cell_ref_err + avg_col_ref_err)
                            ref_strs = ["ang_jmp", "spt_jmp"]
                            ref_str = rng.choice(ref_strs, size = 1, p = (p_ang, 1 - p_ang))[0]
                            #if p_ang >= 0.5:
                            #    ref_str = ref_strs[0]
                            #else:
                            #    ref_str = ref_strs[1]
                            if ref_str == "ang_jmp":
                                err_ind = jmp_ang_err_ind
                            else: # ref_str == "spt_jmp"
                                err_ind = jmp_spt_err_ind
                        if do_plot_err_ind and (trial%10 == 0 or do_calc_err):
                            gen_err_ind_plot(mesh, err_ind_ang, trial, trial_dir, "err_ind_ang.png")
                            gen_err_ind_plot(mesh, err_ind_spt, trial, trial_dir, "err_ind_spt.png")
                        avg_cell_ref_err_str = "{:.4E}".format(jmp_ang_err_ind.avg_cell_ref_err)
                        avg_col_ref_err_str = "{:.4E}".format(jmp_spt_err_ind.avg_col_ref_err)
                        refs += [[ndof, ref_str, avg_cell_ref_err_str, avg_col_ref_err_str, info]]
                    else: # Just refining in space
                        # Calculate nneg errors
                        kwargs_spt_nneg  = combo["kwargs_spt_nneg"]
                        col_ref_tol      = kwargs_spt_nneg["col_ref_tol"]
                        nneg_spt_err_ind = amr.nneg_err_spt(mesh, uh_proj,
                                                            **kwargs_spt_nneg)
                        
                        # If nneg error exceeds tolerance, we refine whichever
                        # has larger negative values in columns to be refined.
                        if (nneg_spt_err_ind.col_max_err < col_ref_tol):
                            err_ind = nneg_spt_err_ind
                            ref_str = "spt_nneg"
                            
                        else:
                            # If not using nneg err, then we use the jump error
                            kwargs_spt_jmp  = combo["kwargs_spt_jmp"]
                            jmp_spt_err_ind = amr.col_jump_err(mesh, uh_proj,
                                                               **kwargs_spt_jmp)
                            err_ind = jmp_spt_err_ind
                            ref_str = "spt_jmp"
                        if do_plot_err_ind and (trial%10 == 0 or do_calc_err):
                            gen_err_ind_plot(mesh, err_ind, trial, trial_dir, "err_ind_spt.png")
                        refs += [[ndof, ref_str, info]]
                    
                    file_name = "refs.txt"
                    file_path = os.path.join(combo_dir, file_name)
                    json.dump(refs, open(file_path, "w"))
                    
                    msg = (
                        "[Trial {}] Refining cause: {}\n".format(trial, ref_str)
                        )
                    utils.print_msg(msg, blocking = False)
                    mesh = amr.ref_by_ind(mesh, err_ind)
                    
            elif ((combo["short_name"] == "h-amr-all")
                  or (combo["short_name"] == "p-amr-all")
                  or (combo["short_name"] == "hp-amr-all")):
                if comm_rank == 0:
                    # Have two choices: Refine based on nneg or jump, and if
                    # also refining in space
                    
                    # Calculate nneg errors
                    kwargs_spt_nneg  = combo["kwargs_spt_nneg"]
                    col_ref_tol      = kwargs_spt_nneg["col_ref_tol"]
                    nneg_spt_err_ind = amr.nneg_err_spt(mesh, uh_proj,
                                                        **kwargs_spt_nneg)
                    
                    kwargs_ang_nneg  = combo["kwargs_ang_nneg"]
                    cell_ref_tol     = kwargs_ang_nneg["cell_ref_tol"]
                    nneg_ang_err_ind = amr.nneg_err_ang(mesh, uh_proj,
                                                        **kwargs_ang_nneg)
                    
                    # If nneg error exceeds tolerance, we refine whichever
                    # has larger negative values in columns to be refined.
                    if ((nneg_ang_err_ind.cell_max_err < cell_ref_tol)
                        or (nneg_spt_err_ind.col_max_err < cell_ref_tol)):
                        # Use cell_ref_tol here because otherwise this
                        # would always trigger.
                        err_ind_ang = nneg_ang_err_ind
                        err_ind_spt = nneg_spt_err_ind

                        avg_cell_ref_err = nneg_ang_err_ind.avg_cell_ref_err
                        avg_col_ref_err  = nneg_spt_err_ind.avg_col_ref_err
                        if np.abs(avg_cell_ref_err) > np.abs(avg_col_ref_err):
                            err_ind = nneg_ang_err_ind
                            ref_str = "ang_nneg"
                        else:
                            err_ind = nneg_spt_err_ind
                            ref_str = "spt_nneg"
                            
                    else:
                        # If not using nneg err, then we use the jump error
                        kwargs_ang_jmp  = combo["kwargs_ang_jmp"]
                        jmp_ang_err_ind = amr.cell_jump_err(mesh, uh_proj,
                                                            **kwargs_ang_jmp)
                        kwargs_spt_jmp  = combo["kwargs_spt_jmp"]
                        jmp_spt_err_ind = amr.col_jump_err(mesh, uh_proj,
                                                           **kwargs_spt_jmp)
                        msg = (
                            "Comparison of average refinement error:\n" +
                            12 * " " + "Angular {:.4E} ".format(jmp_ang_err_ind.avg_cell_ref_err) +
                            "vs. Spatial {:.4E}\n".format(jmp_spt_err_ind.avg_col_ref_err)
                        )
                        utils.print_msg(msg, blocking = False)
                        err_ind_ang = jmp_ang_err_ind
                        err_ind_spt = jmp_spt_err_ind

                        avg_cell_ref_err = jmp_ang_err_ind.avg_cell_ref_err
                        avg_col_ref_err  = jmp_spt_err_ind.avg_col_ref_err
                            
                        # Randomly choose which, but probability depends on steering criterion
                        p_ang = avg_cell_ref_err / (avg_cell_ref_err + avg_col_ref_err)
                        ref_strs = ["ang_jmp", "spt_jmp"]
                        ref_str = rng.choice(ref_strs, size = 1, p = (p_ang, 1 - p_ang))[0]
                        #if p_ang >= 0.5:
                        #    ref_str = ref_strs[0]
                        #else:
                        #    ref_str = ref_strs[1]
                        if ref_str == "ang_jmp":
                            err_ind = jmp_ang_err_ind
                        else: # ref_str == "spt_jmp"
                            err_ind = jmp_spt_err_ind
                    if do_plot_err_ind and (trial%10 == 0 or do_calc_err):
                        gen_err_ind_plot(mesh, err_ind_ang, trial, trial_dir, "err_ind_ang.png")
                        gen_err_ind_plot(mesh, err_ind_spt, trial, trial_dir, "err_ind_spt.png")
                    avg_cell_ref_err_str = "{:.4E}".format(jmp_ang_err_ind.avg_cell_ref_err)
                    avg_col_ref_err_str = "{:.4E}".format(jmp_spt_err_ind.avg_col_ref_err)
                    refs += [[ndof, ref_str, avg_cell_ref_err_str, avg_col_ref_err_str, info]]
                    
                    file_name = "refs.txt"
                    file_path = os.path.join(combo_dir, file_name)
                    json.dump(refs, open(file_path, "w"))
                    
                    msg = (
                        "[Trial {}] Refining cause: {}\n".format(trial, ref_str)
                        )
                    utils.print_msg(msg, blocking = False)
                    mesh = amr.ref_by_ind(mesh, err_ind)
                    
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
            file_path = os.path.join(combo_dir, file_name)
            json.dump(errs, open(file_path, "w"))
            
            file_name = "ndofs.txt"
            file_path = os.path.join(combo_dir, file_name)
            json.dump(ndofs, open(file_path, "w"))
            
            if do_plot_errs:
                fig, ax = plt.subplots()
                
                colors = ["#000000", "#E69F00", "#56B4E9", "#009E73",
                          "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
                          "#882255"]
                
                ax.scatter(ndofs, errs,
                           label = None,
                           color = colors[0])
                
                # Get best-fit line
                [a, b] = np.polyfit(np.log10(ndofs), np.log10(errs), 1)
                xx = np.logspace(np.log10(ndofs[0]), np.log10(ndofs[-1]))
                yy = 10**b * xx**a
                ax.plot(xx, yy,
                        label = "{} High-Res.: {:4.2f}".format(combo_name, a),
                        color = colors[0],
                        linestyle = "--"
                        )
                
                ax.set_xscale("log", base = 10)
                ax.set_yscale("log", base = 10)
                
                err_max = max(errs)
                err_min = min(errs)
                if np.log10(err_max) - np.log10(err_min) < 1:
                    ymin = 10**(np.floor(np.log10(err_min)))
                    ymax = 10**(np.ceil(np.log10(err_max)))
                    ax.set_ylim([ymin, ymax])
                    
                ax.set_xlabel("Total Degrees of Freedom")
                if test_num == 1:
                    ax.set_ylabel(r"$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u - u_{hp} \right)^2\,d\vec{x}\,d\vec{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\vec{s}}}$")
                elif test_num == 2:
                    ax.set_ylabel(r"$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} - u_{hp} \right)^2\,d\vec{x}\,d\vec{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\vec{s}}}$")
                elif test_num == 3:
                    ax.set_ylabel(r"$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} - u_{hp} \right)^2\,d\vec{x}\,d\vec{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\vec{s}}}$")
                elif test_num == 3:
                    ax.set_ylabel(r"$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} - u_{hp} \right)^2\,d\vec{x}\,d\vec{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\vec{s}}}$")
            
                ax.legend()
                
                title_str = ( "{} Convergence Rate".format(combo["full_name"]) )
                ax.set_title(title_str)
                
                file_name = "convergence_{}.png".format(test_num)
                file_path = os.path.join(combo_dir, file_name)
                fig.set_size_inches(6.5, 6.5)
                plt.tight_layout()
                plt.savefig(file_path, dpi = 300)
                plt.close(fig)
                
                combo_ndofs[combo_name] = ndofs
                combo_errs[combo_name]  = errs
                combo_nnz[combo_name]   = mat_info["nz_used"]
                
            perf_combo_f = perf_counter()
            perf_combo_dt = perf_combo_f - perf_combo_0
            msg = (
                "Combination {} complete!\n".format(combo["full_name"]) +
                12 * " " + "Time elapsed: {:08.3f} [s]\n".format(perf_combo_dt)
            )
            utils.print_msg(msg, blocking = False)
            
        # Clear some variables to reduce memory usage
        del mesh, uh_proj
        gc.collect()
        
    if comm_rank == 0:
        if do_plot_errs:
            fig, ax = plt.subplots()
            
            ncombo = len(combos)
            
            colors = ["#000000", "#E69F00", "#56B4E9", "#009E73",
                      "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
                      "#882255"]
            
            for cc in range(0, ncombo):
                combo_name = combo_names[cc]
                ndofs = combo_ndofs[combo_name]
                errs  = combo_errs[combo_name]
                ax.scatter(ndofs, errs,
                           label     = None,
                           color     = colors[cc])
                
                # Get best-fit line
                [a, b] = np.polyfit(np.log10(ndofs), np.log10(errs), 1)
                xx = np.logspace(np.log10(ndofs[0]), np.log10(ndofs[-1]))
                yy = 10**b * xx**a
                ax.plot(xx, yy,
                        label = "{}: {:4.2f}".format(combo_name, a),
                        color = colors[cc],
                        linestyle = "--"
                        )
                
            ax.legend()
            
            ax.set_xscale("log", base = 10)
            ax.set_yscale("log", base = 10)
            
            ax.set_xlabel("Total Degrees of Freedom")
            if test_num == 1:
                ax.set_ylabel(r"$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u - u_{hp} \right)^2\,d\vec{x}\,d\vec{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\vec{s}}}$")
            elif test_num == 2:
                ax.set_ylabel(r"$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} - u_{hp} \right)^2\,d\vec{x}\,d\vec{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\vec{s}}}$")
            elif test_num == 3:
                ax.set_ylabel(r"$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} - u_{hp} \right)^2\,d\vec{x}\,d\vec{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\vec{s}}}$")
            elif test_num == 3:
                ax.set_ylabel(r"$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} - u_{hp} \right)^2\,d\vec{x}\,d\vec{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\vec{s}}}$")
                
            title_str = ( "Convergence Rate" )
            ax.set_title(title_str)
            
            file_name = "convergence_{}.png".format(test_num)
            file_path = os.path.join(figs_dir, file_name)
            fig.set_size_inches(6.5, 6.5)
            plt.tight_layout()
            plt.savefig(file_path, dpi = 300)
            plt.close(fig)
            
        perf_all_f = perf_counter()
        perf_all_dt = perf_all_f - perf_all_0
        msg = (
            "Test {} figures generated!\n".format(test_num) +
            12 * " " + "Time elapsed: {:08.3f} [s]\n".format(perf_all_dt)
        )
        utils.print_msg(msg)
        
def gen_kappa_sigma_plots(Ls, kappa, sigma, figs_dir, file_names):
    [Lx, Ly] = Ls[:]
    
    xx = np.linspace(0, Lx, num = 1000).reshape([1, 1000])
    yy = np.linspace(0, Ly, num = 1000).reshape([1000, 1])
    [XX, YY] = np.meshgrid(xx, yy)
    
    kappa_c = kappa(xx, yy)
    sigma_c = sigma(xx, yy)
    [vmin, vmax] = [0., max(np.amax(kappa_c), np.amax(sigma_c))]
    
    cmap = mpl.cm.gray
    norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
    
    # kappa Plot
    file_path = os.path.join(figs_dir, file_names[0])
    if not os.path.isfile(file_path):
        fig, ax = plt.subplots()
        
        kappa_plot = ax.contourf(XX, YY, kappa_c, levels = 16,
                                 cmap = cmap, norm = norm)
        
        ax.set_xlim([0, Lx])
        ax.set_ylim([0, Ly])
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(r"$\kappa\left( x,\ y \right)$")
        
        fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap), ax = ax)
        
        file_path = os.path.join(figs_dir, file_names[0])
        plt.tight_layout()
        plt.savefig(file_path, dpi = 300)
        
        plt.close(fig)

    # sigma Plot
    file_path = os.path.join(figs_dir, file_names[1])
    if not os.path.isfile(file_path):
        fig, ax = plt.subplots()
        
        kappa_plot = ax.contourf(XX, YY, sigma_c, levels = 16,
                                 cmap = cmap, norm = norm)
        
        ax.set_xlim([0, Lx])
        ax.set_ylim([0, Ly])
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(r"$\sigma\left( x,\ y \right)$")
        
        fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap), ax = ax)
        
        plt.tight_layout()
        plt.savefig(file_path, dpi = 300)
        
        plt.close(fig)
    
def gen_Phi_plot(Phi, figs_dir, file_name):
    
    file_path = os.path.join(figs_dir, file_name)
    if not os.path.isfile(file_path):
        th = np.linspace(0, 2. * np.pi, num = 720)
        rr = Phi(0, th)
        
        max_r = np.amax(rr)
        ntick = 2
        r_ticks = np.linspace(max_r / ntick, max_r, ntick)
        r_tick_labels = ["{:3.2f}".format(r_tick) for r_tick in r_ticks]
        th_ticks = np.linspace(0, 2. * np.pi, num = 8, endpoint = False)
        th_tick_labels = [r"${:3.2f} \pi$".format(th_tick/np.pi)
                          for th_tick in th_ticks]
        
        fig, ax = plt.subplots(subplot_kw = {"projection": "polar"})
        
        Phi_plot = ax.plot(th, rr, color = "black")
        
        ax.set_rlim([0, max_r])
        ax.set_rticks(r_ticks, r_tick_labels)
        ax.set_xlabel(r"$\theta - \theta"$")
        ax.set_xticks(th_ticks, th_tick_labels)
        ax.set_title(r"$\Phi\left( \theta - \theta" \right)$")
        
        plt.tight_layout()
        plt.savefig(file_path, dpi = 300)
        
        plt.close(fig)
    
def gen_u_plot(Ls, u, figs_dir):
    perf_0 = perf_counter()
    msg = ( "Plotting analytic solution...\n"
           )
    utils.print_msg(msg, blocking = False)

    mesh = ji_mesh.Mesh(Ls     = Ls[:],
                        pbcs   = [False, False],
                        ndofs  = [8, 8, 8],
                        has_th = True)
    for _ in range(0, 4):
        mesh.ref_mesh(kind = "ang", form = "h")
    for _ in range(0, 4):
        mesh.ref_mesh(kind = "spt", form = "h")
        
    file_names = ["u_th.png", "u_xy.png", "u_xth.png", "u_yth.png", "u_xyth.png"]
    file_paths = []
    is_file_paths = []
    for file_name in file_names:
        file_path      = os.path.join(figs_dir, file_name)
        file_paths    += [file_path]
        is_file_paths += [os.path.isfile(file_path)]
        
    if not all(is_file_paths):
        u_proj = proj.Projection(mesh, u)
        
        if not os.path.isfile(file_paths[0]):
            proj.utils.plot_th(mesh, u_proj, file_name = file_paths[0])
            
        if not os.path.isfile(file_paths[1]):
            proj.utils.plot_xy(mesh, u_proj, file_name = file_paths[1])
            
        if not os.path.isfile(file_paths[2]):
            proj.utils.plot_xth(mesh, u_proj, file_name = file_paths[2])
            
        if not os.path.isfile(file_paths[3]):
            proj.utils.plot_yth(mesh, u_proj, file_name = file_paths[3])
            
        if not os.path.isfile(file_paths[4]):
            proj.utils.plot_xyth(mesh, u_proj, file_name = file_paths[4])
    
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = ( "Analytic solution plotted!\n" +
            12 * " " + "Time Elapsed: {:08.3f} [s]\n".format(perf_diff)
           )
    utils.print_msg(msg, blocking = False)
    
    
def get_soln(mesh, kappa, sigma, Phi, bcs_dirac, f, trial, **kwargs):
    perf_0 = perf_counter()
    msg = (
        "[Trial {}] Obtaining numerical solution...\n".format(trial)
    )
    utils.print_msg(msg)
    
    [uh_proj, info, mat_info] = rt.rtdg(mesh, kappa, sigma, Phi, bcs_dirac, f,
                                  verbose = True, **kwargs)
    PETSc.garbage_cleanup()
    
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = (
        "[Trial {}] Numerical solution obtained! : ".format(trial) +
        "Exit Code {} \n".format(info) +
        22 * " " + "Time Elapsed: {:08.3f} [s]\n".format(perf_diff)
    )
    utils.print_msg(msg)

    return [uh_proj, info, mat_info]

def get_err(mesh, uh_proj, u, kappa, sigma, Phi, bcs_dirac, f,
            trial, figs_dir, **kwargs):
    default_kwargs = {"res_coeff" : 1,
                      "err_kind" : "anl"}
    kwargs = {**default_kwargs, **kwargs}
    
    perf_0 = perf_counter()
    msg = ( "[Trial {}] Obtaining error...\n".format(trial)
           )
    utils.print_msg(msg)

    if kwargs["err_kind"] == "anl":
        err = amr.total_anl_err(mesh, uh_proj, u, **kwargs)
    elif kwargs["err_kind"] == "hr":
        err = amr.high_res_err(mesh, uh_proj,
                               kappa, sigma, Phi, bcs_dirac, f,
                               verbose = True,
                               dir_name = figs_dir,
                               **kwargs)
    
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = (
        "[Trial {}] Error obtained! : {:.4E}\n".format(trial, err) +
        22 * " " + "Time Elapsed: {:08.3f} [s]\n".format(perf_diff)
    )
    utils.print_msg(msg)
    
    return err

def gen_mesh_plot(mesh, trial, trial_dir, **kwargs):
    
    default_kwargs = {"blocking" : False # Default to non-blocking behavior for plotting
                      }
    kwargs = {**default_kwargs, **kwargs}

    perf_0 = perf_counter()
    msg = ( "[Trial {}] Plotting mesh...\n".format(trial)
           )
    utils.print_msg(msg, **kwargs)
    
    file_name = "mesh_3d_{}.png".format(trial)
    file_path = os.path.join(trial_dir, file_name)
    ji_mesh.utils.plot_mesh(mesh      = mesh,
                            file_name = file_path,
                            plot_dim  = 3)
    
    file_name = "mesh_2d_{}.png".format(trial)
    file_path = os.path.join(trial_dir, file_name)
    ji_mesh.utils.plot_mesh(mesh        = mesh,
                            file_name   = file_path,
                            plot_dim    = 2,
                            label_cells = (trial <= 2))
    
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = ( "[Trial {}] Mesh plotted!\n".format(trial) +
            22 * " " + "Time Elapsed: {:08.3f} [s]\n".format(perf_diff)
           )
    utils.print_msg(msg, **kwargs)

def gen_mesh_plot_p(mesh, trial, trial_dir, **kwargs):

    default_kwargs = {"blocking" : False # Default to non-blocking behavior for plotting
                      }
    kwargs = {**default_kwargs, **kwargs}
    
    perf_0 = perf_counter()
    msg = ( "[Trial {}] Plotting mesh polynomial degree...\n".format(trial)
           )
    utils.print_msg(msg, **kwargs)
    
    file_name = "mesh_3d_p_{}.png".format(trial)
    file_path = os.path.join(trial_dir, file_name)
    ji_mesh.utils.plot_mesh_p(mesh        = mesh,
                              file_name   = file_path,
                              plot_dim    = 3)
    
    file_name = "mesh_2d_p_{}.png".format(trial)
    file_path = os.path.join(trial_dir, file_name)
    ji_mesh.utils.plot_mesh_p(mesh        = mesh,
                              file_name   = file_path,
                              plot_dim    = 2,
                              label_cells = (trial <= 3))
    
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = ( "[Trial {}] Mesh polynomial degree plotted!\n".format(trial) +
            22 * " " + "Time Elapsed: {:08.3f} [s]\n".format(perf_diff)
           )
    utils.print_msg(msg, **kwargs)
    
def gen_uh_plot(mesh, uh_proj, trial, trial_dir, **kwargs):
    
    default_kwargs = {"blocking" : False # Default to non-blocking behavior for plotting
                      }
    kwargs = {**default_kwargs, **kwargs}
    
    perf_0 = perf_counter()
    msg = (
        "[Trial {}] Plotting numerical solution...\n".format(trial)
    )
    utils.print_msg(msg, **kwargs)
    
    file_name = "uh_th_{}.png".format(trial)
    file_path = os.path.join(trial_dir, file_name)
    proj.utils.plot_th(mesh, uh_proj, file_name = file_path)
    
    file_name = "uh_xy_{}.png".format(trial)
    file_path = os.path.join(trial_dir, file_name)
    proj.utils.plot_xy(mesh, uh_proj, file_name = file_path)
    
    file_name = "uh_xth_{}.png".format(trial)
    file_path = os.path.join(trial_dir, file_name)
    proj.utils.plot_xth(mesh, uh_proj, file_name = file_path)
    
    file_name = "uh_yth_{}.png".format(trial)
    file_path = os.path.join(trial_dir, file_name)
    proj.utils.plot_yth(mesh, uh_proj, file_name = file_path)
    
    file_name = "uh_xyth_{}.png".format(trial)
    file_path = os.path.join(trial_dir, file_name)
    proj.utils.plot_xyth(mesh, uh_proj, file_name = file_path)
    
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = (
        "[Trial {}] Numerical solution plotted!\n".format(trial) +
        22 * " " + "Time Elapsed: {:08.3f} [s]\n".format(perf_diff)
    )
    utils.print_msg(msg, **kwargs)
    
def gen_err_ind_plot(mesh, err_ind, trial, trial_dir, file_name, **kwargs):
    
    default_kwargs = {"blocking" : False # Default to non-blocking behavior for plotting
                      }
    kwargs = {**default_kwargs, **kwargs}
    
    perf_0 = perf_counter()
    msg = (
        "[Trial {}] Plotting error indicators...\n".format(trial)
    )
    utils.print_msg(msg, **kwargs)
    
    file_path = os.path.join(trial_dir, file_name)
    amr.utils.plot_error_indicator(mesh, err_ind, file_name = file_path)
    if err_ind.ref_cell:
        file_path = os.path.join(trial_dir, "cell_jumps.png")
        amr.utils.plot_cell_jumps(mesh, err_ind, file_name = file_path)
    
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = (
        "[Trial {}] Error indicator plotted!\n".format(trial) +
        22 * " " + "Time Elapsed: {:08.3f} [s]\n".format(perf_diff)
    )
    utils.print_msg(msg, **kwargs)

if __name__ == "__main__":
    main()