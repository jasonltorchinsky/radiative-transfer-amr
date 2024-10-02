import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve, eigs
from time import perf_counter
import os, sys

from .gen_mesh import gen_mesh

sys.path.append("../../tests")
from test_cases import get_cons_prob

sys.path.append("../../src")
from dg.mesh.utils import plot_mesh
from dg.matrix import get_intr_mask, split_matrix, merge_vectors
from dg.projection import Projection, push_forward, to_projection
from dg.projection.utils import plot_projection
import dg.quadrature as qd
from rt import calc_mass_matrix, calc_scat_matrix, \
    calc_intr_conv_matrix, calc_bdry_conv_matrix, \
    calc_forcing_vec
from amr import intg_col_bdry_th

from utils import print_msg


def test_0(dir_name = "test_amr"):
    """
    Tests the angular integration on the spatial boundary of columns.
    """
    
    test_dir = os.path.join(dir_name, "test_0")
    os.makedirs(test_dir, exist_ok = True)

    # Set the refinement type: "sin" - single column
    #                        : "uni" - uniform
    #                        : "amr" - adaptive
    ref_type = "uni"
    ntrial   = 3
    
    # Get the base mesh, test_problem
    [Lx, Ly]                   = [3., 2.]
    pbcs                       = [True, False]
    [ndof_x, ndof_y, ndof_th]  = [2, 2, 2]
    has_th                     = True
    mesh = gen_mesh(Ls     = [Lx, Ly],
                    pbcs   = pbcs,
                    ndofs  = [ndof_x, ndof_y, ndof_th],
                    has_th = has_th)
    
    [anl_sol, kappa, sigma, Phi, f, anl_sol_intg_th] = get_cons_soln(prob_name = "comp",
                                                                     sol_num   = 0)
    
    # Solve simplified problem over several trials
    ref_ndofs = np.zeros([ntrial])
    sol_errs  = np.zeros([ntrial])
    intg_errs = np.zeros([ntrial])
    for trial in range(0, ntrial):
        perf_trial_0 = perf_counter()
        print_msg("[Trial {}] Starting...".format(trial))
            
        # Set up output directories
        trial_dir = os.path.join(test_dir, "trial_{}".format(trial))
        os.makedirs(trial_dir, exist_ok = True)

        # Plot the mesh
        file_name = os.path.join(trial_dir, "mesh_3d.png")
        plot_mesh(mesh,
                  file_name = file_name,
                  plot_dim  = 3)
        file_name = os.path.join(trial_dir, "mesh_2d.png")
        plot_mesh(mesh,
                  file_name   = file_name,
                  plot_dim    = 2,
                  label_cells = (trial <= 3))

        # Plot the coefficient functions
        
        # Construct solve the test problem
        perf_cons_0 = perf_counter()
        print_msg("[Trial {}] Solving the test problem...".format(trial))

        # Use the analytic solution for boundary conditions
        M_mass = calc_mass_matrix(mesh, kappa)
        M_scat = calc_scat_matrix(mesh, sigma, Phi)
        M_intr_conv = calc_intr_conv_matrix(mesh)
        M_bdry_conv = calc_bdry_conv_matrix(mesh)


        f_vec       = get_forcing_vec(mesh, f)
        anl_sol_vec = get_projection_vec(mesh, anl_sol)
        
        intr_mask        = get_intr_mask(mesh)
        bdry_mask        = np.invert(intr_mask)
        f_vec_intr       = f_vec[intr_mask]
        anl_sol_vec_intr = anl_sol_vec[intr_mask]
        bcs_vec          = anl_sol_vec[bdry_mask]
        
        M = (M_bdry_conv - M_intr_conv) + M_mass - M_scat
        [M_intr, M_bdry] = split_matrix(mesh, M, intr_mask)
        
        u_vec_intr = spsolve(M_intr, f_vec_intr - M_bdry @ bcs_vec)
        u_vec      = merge_vectors(u_vec_intr, bcs_vec, intr_mask)
        u_proj     = to_projection(mesh, u_vec)
        
        perf_cons_f    = perf_counter()
        perf_cons_diff = perf_cons_f - perf_cons_0
        msg = (
            "[Trial {}] Test problem solved! ".format(trial) +
            "Time Elapsed: {:08.3f} [s]".format(perf_cons_diff)
        )
        print_msg(msg)

        # Get number of DOFs
        ref_ndofs[trial] = np.size(f_vec)

        # Calculate the maximum error of the solution
        anl_sol_proj = Projection(mesh, anl_sol)
        anl_sol_vec  = anl_sol_proj.to_vector()
        diff_vec     = np.abs(u_vec - anl_sol_vec)
        max_err      = np.amax(diff_vec)
        
        sol_errs[trial] = max_err

        # Calculate the error of angular integration
        max_err      = 0.
        proj_col_items = sorted(u_proj.cols.items())
        for col_key, proj_col in proj_col_items:
            if proj_col.is_lf:
                col_intg_th = intg_col_bdry_th(mesh, u_proj, col_key)

                [x0, y0, xf, yf] = proj_col.pos[:]
                [dx, dy]         = [xf - x0, yf - y0]
                [ndof_x, ndof_y] = proj_col.ndofs[:]

                [xxb, _, yyb, _, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                      nnodes_y = ndof_y)

                xxf = push_forward(x0, xf, xxb)
                yyf = push_forward(y0, yf, yyb)

                for F in range(0, 4):
                    if (F%2 == 0):
                        if (F == 0):
                            x_idx = ndof_x - 1
                        elif (F == 2):
                            x_idx = 0

                        x_i = xxf[x_idx]
                        for jj in range(0, ndof_y):
                            y_j = yyf[jj]
                            err = np.abs(col_intg_th[F][jj] - anl_sol_intg_th(x_i, y_j))
                            max_err = max(max_err, err)
                    
                    else:
                        if (F == 1):
                            y_idx = ndof_y - 1
                        elif (F == 3):
                            y_idx = 0

                        y_j = yyf[y_idx]
                        for ii in range(0, ndof_x):
                            x_i = xxf[ii]
                            err = np.abs(col_intg_th[F][ii] - anl_sol_intg_th(x_i, y_j))
                            max_err = max(max_err, err)

        msg = (
            "[Trial {}] Max integration error: {} ".format(trial, max_err)
        )
        print_msg(msg)
        
        intg_errs[trial] = max_err
        
        # Refine the mesh for the next trial
        if ref_type == "sin":
            ## Refine a given column
            col_keys = sorted(mesh.cols.keys())
            mesh.ref_col(col_keys[-4], kind = "all")
        elif ref_type == "uni":
            ## Refine the mesh uniformly
            mesh.ref_mesh(kind = "spt")
            
        perf_trial_f    = perf_counter()
        perf_trial_diff = perf_trial_f - perf_trial_0
        msg = (
            "[Trial {}] Trial completed! ".format(trial) +
            "Time Elapsed: {:08.3f} [s]\n".format(perf_trial_diff)
        )
        print_msg(msg)
        
    # Plot solution errors
    fig, ax = plt.subplots()
    
    ax.plot(ref_ndofs, sol_errs,
            label     = "L$^{\infty}$ Error",
            color     = "k",
            linestyle = "-")

    ax.set_xscale("log", base = 2)
    ax.set_yscale("log", base = 2)

    if np.log2(max(sol_errs)) - np.log2(min(sol_errs)) < 1:
        ymin = 2**(np.floor(np.log2(min(sol_errs))))
        ymax = 2**(np.ceil(np.log2(max(sol_errs))))
        ax.set_ylim([ymin, ymax])
    
    ax.set_xlabel("Total Degrees of Freedom")
    ax.set_ylabel("L$^{\infty}$ Error")


    ref_str = ""
    if ref_type == "sin":
        ref_str = "Single Column"
    elif ref_type == "uni":
        ref_str = "Uniform"
    elif ref_type == "amr":
        ref_str = "Adaptive"
    title_str = "{} $h$-Refinement Solution Convergence Rate".format(ref_str)
    ax.set_title(title_str)
    
    file_name = "h-sol-convergence.png"
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(test_dir, file_name), dpi = 300)
    plt.close(fig)


    # Plot integration errors
    fig, ax = plt.subplots()
    
    ax.plot(ref_ndofs, intg_errs,
            label     = "L$^{\infty}$ Error",
            color     = "k",
            linestyle = "-")

    ax.set_xscale("log", base = 2)
    ax.set_yscale("log", base = 2)

    if np.log2(max(intg_errs)) - np.log2(min(intg_errs)) < 1:
        ymin = 2**(np.floor(np.log2(min(intg_errs))))
        ymax = 2**(np.ceil(np.log2(max(intg_errs))))
        ax.set_ylim([ymin, ymax])
    
    ax.set_xlabel("Total Degrees of Freedom")
    ax.set_ylabel("L$^{\infty}$ Error")

    ref_str = ""
    if ref_type == "sin":
        ref_str = "Single Column"
    elif ref_type == "uni":
        ref_str = "Uniform"
    elif ref_type == "amr":
        ref_str = "Adaptive"
    title_str = "{} $h$-Refinement Angular-Integral Convergence Rate".format(ref_str)
    ax.set_title(title_str)
    
    file_name = "h-intg-convergence.png"
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(test_dir, file_name), dpi = 300)
    plt.close(fig)
