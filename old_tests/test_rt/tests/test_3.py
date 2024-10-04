import json
import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
import os
import petsc4py
import psutil
import sys
from   mpi4py   import MPI
from   petsc4py import PETSc
from   time     import perf_counter

import dg.matrix     as mat
import dg.mesh       as ji_mesh
import dg.projection as proj
import dg.quadrature as qd
import rt
import amr
import utils
from   test_cases    import get_cons_prob, h_uni_ang, p_uni_ang, hp_uni_ang, \
    h_uni_spt, p_uni_spt, hp_uni_spt, h_uni_all, p_uni_all, hp_uni_all, \
    h_amr_ang, p_amr_ang, hp_amr_ang, h_amr_spt, p_amr_spt, hp_amr_spt, \
    h_amr_all, p_amr_all, hp_amr_all

def test_3(dir_name = "test_rt"):
    """
    Solves constructed ("manufactured") problems, with options for different
    types of refinement.
    """
    
    petsc4py.init()
    
    # MPI COMM for communicating data
    MPI_comm = MPI.COMM_WORLD
    
    #PETSc COMM for parallel matrix construction, solves
    comm = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()
    
    test_dir = os.path.join(dir_name, "test_3")
    os.makedirs(test_dir, exist_ok = True)

    ## Test stopping parameters
    # Maximum number of DOFs
    max_ndof = 2**17
    # Maximum number of trials
    max_ntrial = 16
    # Minimum Error
    min_err = 1.e-7
    # Which combinations of Refinement Form, Refinement Type, and Refinement Kind
    combos = [
        hp_amr_ang,
        hp_amr_spt,
        hp_amr_all
    ]
    
    # Test Output Parameters
    do_plot_mesh        = True
    do_plot_mesh_p      = True
    do_plot_uh          = True
    do_plot_u           = True
    do_plot_diff        = True
    do_plot_err_ind     = True
    do_plot_errs        = True

    # Which problems to solve
    prob_nums = []
    for x_num in range(2, 4):
        for y_num in range(2, 4):
            for th_num in range(3, 4):
                for scat_num in range(0, 3):
                    prob_nums += [[x_num, y_num, th_num, scat_num]]
                    
    for prob_num in [[2, 2, 3, 2]]:
        prob_dir = os.path.join(test_dir, str(prob_num))
        os.makedirs(prob_dir, exist_ok = True)
        
        msg = (
            "Starting problem {}...\n".format(prob_num)
        )
        utils.print_msg(msg)
        
        combo_ndofs = {}
        combo_errs  = {}
        
        for combo in combos:
            combo_str  = combo["short_name"]
            combo_name = combo_str
            combo_dir  = os.path.join(prob_dir, combo_str)
            os.makedirs(combo_dir, exist_ok = True)
            
            perf_combo_0 = perf_counter()
            msg = ( "Starting combination {}...\n".format(combo_str) )
            utils.print_msg(msg)
            
            # Get the base mesh, manufactured solution
            [Lx, Ly]                   = [2., 3.]
            pbcs                       = [False, False]
            mesh = ji_mesh.Mesh(Ls     = [Lx, Ly],
                                pbcs   = pbcs,
                                ndofs  = combo["ndofs"],
                                has_th = True)
            
            for _ in range(0, combo["nref_ang"]):
                mesh.ref_mesh(kind = "ang", form = "h")
                
            for _ in range(0, combo["nref_spt"]):
                mesh.ref_mesh(kind = "spt", form = "h")
            
            [u, kappa, sigma, Phi, f,
             u_intg_th, u_intg_xy] = get_cons_prob(prob_name = "comp",
                                                   prob_num  = prob_num,
                                                   mesh      = mesh,
                                                   sth       = 64.)
            bcs_dirac = [u, [None, None, None]]
            
            if comm_rank == 0:
                kappa_file_name = "kappa_{}.png".format(prob_num[0:3])
                sigma_file_name = "sigma_{}.png".format(prob_num[0:3])
                Phi_file_name   = "Phi_{}.png".format(prob_num[3])
                gen_kappa_sigma_plots([Lx, Ly], kappa, sigma, prob_dir,
                                      [kappa_file_name, sigma_file_name])
                gen_Phi_plot(Phi, prob_dir, Phi_file_name)
                gen_u_plot([Lx, Ly], u, prob_dir)
                
            MPI_comm.Barrier()
            
            # Solve the manufactured problem over several trials
            ndofs = []
            errs  = []
            
            if comm_rank == 0:
                ndof = mesh.get_ndof()
                ndof = MPI_comm.bcast(ndof, root = 0)
            else:
                ndof = None
                ndof = MPI_comm.bcast(ndof, root = 0)
            trial = 0
            err   = 1.
            mem_used = psutil.virtual_memory()[2]
            while (((ndof < max_ndof)
                    and (trial <= max_ntrial)
                    and (err > min_err)
                    and (mem_used <= 95.))
                   or (trial <= 1)):
                
                mem_used = psutil.virtual_memory()[2]
                if comm_rank == 0:
                    ndof = mesh.get_ndof()
                    ndof = MPI_comm.bcast(ndof, root = 0)
                else:
                    ndof = None
                    ndof = MPI_comm.bcast(ndof, root = 0)
                ndofs += [ndof]
                
                perf_trial_0 = perf_counter()
                msg = (
                    "[Trial {}] Starting with: ".format(trial) +
                    "{} of {} DoFs and\n".format(ndof, max_ndof) +
                    37 * " " + "error {:.2E} of {:.2E}\n".format(err, min_err) +
                    37 * " " + "RAM Memory % Used: {}\n".format(mem_used)
                )
                utils.print_msg(msg)
                
                # Set up output directories
                trial_dir = os.path.join(combo_dir, "trial_{}".format(trial))
                os.makedirs(trial_dir, exist_ok = True)
                
                uh_proj = get_soln(mesh, kappa, sigma, Phi, bcs_dirac, f,
                                   trial)
                err     = get_err(mesh, uh_proj, u, kappa, sigma, Phi,
                                  bcs_dirac, f,
                                  trial, trial_dir,
                                  nref_ang  = combo["nref_ang"],
                                  nref_spt  = combo["nref_spt"],
                                  ref_kind  = combo["ref_kind"],
                                  res_coeff = 1,
                                  key       = tuple(prob_num))
                errs += [err]
                
                if comm_rank == 0:
                    # Write error results to files as we go along
                    file_name = "errs.txt"
                    file_path = os.path.join(combo_dir, file_name)
                    json.dump(errs, open(file_path, "w"))
                    
                    file_name = "ndofs.txt"
                    file_path = os.path.join(combo_dir, file_name)
                    json.dump(ndofs, open(file_path, "w"))
                    
                    if do_plot_mesh:
                        gen_mesh_plot(mesh, trial, trial_dir, blocking = False)
                        
                    if do_plot_mesh_p:
                        gen_mesh_plot_p(mesh, trial, trial_dir, blocking = False)
                        
                    if do_plot_uh:
                        gen_uh_plot(mesh, uh_proj, trial, trial_dir, blocking = False)
                        
                if   combo["short_name"] == "h-uni-ang":
                    mesh.ref_mesh(kind = "ang", form = "h")
                elif combo["short_name"] == "p-uni-ang":
                    for _ in range(0, 3):
                        mesh.ref_mesh(kind = "ang", form = "p")
                elif combo["short_name"] == "hp-uni-ang":
                    for _ in range(0, 2):
                        mesh.ref_mesh(kind = "ang", form = "p")
                    mesh.ref_mesh(kind = "ang", form = "h")
                elif combo["short_name"] == "h-uni-spt":
                    mesh.ref_mesh(kind = "spt", form = "h")
                elif combo["short_name"] == "p-uni-spt":
                    for _ in range(0, 3):
                        mesh.ref_mesh(kind = "spt", form = "p")
                elif combo["short_name"] == "hp-uni-spt":
                    for _ in range(0, 2):
                        mesh.ref_mesh(kind = "spt", form = "p")
                    mesh.ref_mesh(kind = "spt", form = "h")
                elif combo["short_name"] == "h-uni-all":
                    mesh.ref_mesh(kind = "all", form = "h")
                elif combo["short_name"] == "p-uni-all":
                    for _ in range(0, 3):
                        mesh.ref_mesh(kind = "all", form = "p")
                elif combo["short_name"] == "hp-uni-all":
                    for _ in range(0, 2):
                        mesh.ref_mesh(kind = "all", form = "p")
                    mesh.ref_mesh(kind = "all", form = "h")
                elif ((combo["short_name"] == "h-amr-ang")
                      or (combo["short_name"] == "p-amr-ang")
                      or (combo["short_name"] == "hp-amr-ang")):
                    if comm_rank == 0:
                        uh_vec = uh_proj.to_vector()
                        if np.any(uh_vec < combo["kwargs_ang_nneg"]["cell_ref_tol"]):
                            kwargs_ang  = combo["kwargs_ang_nneg"]
                            err_ind_ang = amr.nneg_err(mesh, uh_proj, **kwargs_ang)
                        else:
                            kwargs_ang  = combo["kwargs_ang_jmp"]
                            err_ind_ang = amr.cell_jump_err(mesh, uh_proj, **kwargs_ang)
                        if do_plot_err_ind:
                            gen_err_ind_plot(mesh, err_ind_ang, trial, trial_dir, "err_ind_ang.png")
                        mesh = amr.ref_by_ind(mesh, err_ind_ang)
                elif ((combo["short_name"] == "h-amr-spt")
                      or (combo["short_name"] == "p-amr-spt")
                      or (combo["short_name"] == "hp-amr-spt")):
                    if comm_rank == 0:
                        uh_vec = uh_proj.to_vector()
                        if np.any(uh_vec < combo["kwargs_spt_nneg"]["col_ref_tol"]):
                            kwargs_spt  = combo["kwargs_spt_nneg"]
                            err_ind_spt = amr.nneg_err(mesh, uh_proj, **kwargs_spt)
                        else:
                            kwargs_spt  = combo["kwargs_spt_jmp"]
                            err_ind_spt = amr.col_jump_err(mesh, uh_proj, **kwargs_spt)
                        if do_plot_err_ind:
                            gen_err_ind_plot(mesh, err_ind_spt, trial, trial_dir, "err_ind_spt.png")
                        mesh = amr.ref_by_ind(mesh, err_ind_spt)
                elif ((combo["short_name"] == "h-amr-all")
                      or (combo["short_name"] == "p-amr-all")
                      or (combo["short_name"] == "hp-amr-all")):
                    if comm_rank == 0:
                        uh_vec = uh_proj.to_vector()
                        if np.any(uh_vec < combo["kwargs_ang_nneg"]["cell_ref_tol"]):
                            kwargs_ang  = combo["kwargs_ang_nneg"]
                            err_ind_ang = amr.nneg_err(mesh, uh_proj, **kwargs_ang)
                        else:
                            kwargs_ang  = combo["kwargs_ang_jmp"]
                            err_ind_ang = amr.cell_jump_err(mesh, uh_proj, **kwargs_ang)
                        if np.any(uh_vec < combo["kwargs_spt_nneg"]["col_ref_tol"]):
                            kwargs_spt  = combo["kwargs_spt_nneg"]
                            err_ind_spt = amr.nneg_err(mesh, uh_proj, **kwargs_spt)
                        else:
                            kwargs_spt  = combo["kwargs_spt_jmp"]
                            err_ind_spt = amr.col_jump_err(mesh, uh_proj, **kwargs_spt)
                        if do_plot_err_ind:
                            gen_err_ind_plot(mesh, err_ind_ang, trial, trial_dir, "err_ind_ang.png")
                            gen_err_ind_plot(mesh, err_ind_spt, trial, trial_dir, "err_ind_spt.png")
                        mesh = amr.ref_by_ind(mesh, err_ind_ang)
                        mesh = amr.ref_by_ind(mesh, err_ind_spt)
                        
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
                # Write error results to files
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
                    anl_err_str = (
                        r"$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u - u_{hp} \right)^2\,d\vec{x}\,d\hat{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\hat{s}}}$"
                    )
                    ax.set_ylabel(anl_err_str)
                    
                    ax.legend()
                    
                    title_str = ( "{} Convergence Rate".format(combo["full_name"]) )
                    ax.set_title(title_str)
                    
                    file_name = "convergence.png"
                    file_path = os.path.join(combo_dir, file_name)
                    fig.set_size_inches(6.5, 6.5)
                    plt.tight_layout()
                    plt.savefig(file_path, dpi = 300)
                    plt.close(fig)
                    
            combo_ndofs[combo_name] = ndofs
            combo_errs[combo_name]  = errs
            
            perf_combo_f = perf_counter()
            perf_combo_dt = perf_combo_f - perf_combo_0
            msg = (
                "Combination {} complete!\n".format(combo["full_name"]) +
                12 * " " + "Time elapsed: {:08.3f} [s]\n".format(perf_combo_dt)
            )
            utils.print_msg(msg)
            
        if comm_rank == 0:
            # Write error results to files
            file_name = "errs.txt"
            file_path = os.path.join(prob_dir, file_name)
            json.dump(combo_errs, open(file_path, "w"))
            
            file_name = "ndofs.txt"
            file_path = os.path.join(prob_dir, file_name)
            json.dump(combo_ndofs, open(file_path, "w"))
            
            if do_plot_errs:
                fig, ax = plt.subplots()
                
                ncombo = len(combos)
                
                colors = ["#000000", "#E69F00", "#56B4E9", "#009E73",
                          "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
                          "#882255"]
                
                combo_names = list(combo_ndofs.keys())
                
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
                anl_err_str = (
                    r"$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u - u_{hp} \right)^2\,d\vec{x}\,d\hat{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\hat{s}}}$"
                )
                ax.set_ylabel(anl_err_str)
                
                title_str = ( "Convergence Rate" )
                ax.set_title(title_str)
                
                file_name = "convergence.png"
                file_path = os.path.join(prob_dir, file_name)
                fig.set_size_inches(6.5, 6.5)
                plt.tight_layout()
                plt.savefig(file_path, dpi = 300)
                plt.close(fig)
                
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
    
    
def get_soln(mesh, kappa, sigma, Phi, bcs_dirac, f, trial):
    perf_0 = perf_counter()
    msg = (
        "[Trial {}] Obtaining numerical solution...\n".format(trial)
    )
    utils.print_msg(msg)

    [uh_proj, info] = rt.rtdg(mesh, kappa, sigma, Phi, bcs_dirac, f,
                              verbose = True)
        
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = (
        "[Trial {}] Numerical solution obtained! : ".format(trial) +
        "Exit Code {} \n".format(info) +
        22 * " " + "Time Elapsed: {:08.3f} [s]\n".format(perf_diff)
    )
    utils.print_msg(msg)

    return uh_proj

def get_err(mesh, uh_proj, u, kappa, sigma, Phi, bcs_dirac, f,
            trial, figs_dir, **kwargs):
    default_kwargs = {"res_coeff" : 1}
    kwargs = {**default_kwargs, **kwargs}
    
    perf_0 = perf_counter()
    msg = ( "[Trial {}] Obtaining error...\n".format(trial)
           )
    utils.print_msg(msg)
    
    err = amr.total_anl_err(mesh, uh_proj, u, **kwargs)
    
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
    
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = (
        "[Trial {}] Error indicator plotted!\n".format(trial) +
        22 * " " + "Time Elapsed: {:08.3f} [s]\n".format(perf_diff)
    )
    utils.print_msg(msg, **kwargs)
