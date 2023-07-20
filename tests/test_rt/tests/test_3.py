import matplotlib        as mpl
import matplotlib.pyplot as plt
import numpy             as np
import os
import petsc4py
import sys
from   mpi4py   import MPI
from   petsc4py import PETSc
from   time     import perf_counter

from test_cases import get_cons_prob, h_uni_ang

import dg.matrix     as mat
import dg.mesh       as ji_mesh
import dg.projection as proj
import dg.quadrature as qd
import rt
import amr
import utils

def test_3(dir_name = 'test_rt'):
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
    
    test_dir = os.path.join(dir_name, 'test_3')
    os.makedirs(test_dir, exist_ok = True)

    ## Test stopping parameters
    # Maximum number of DOFs
    max_ndof = 2**16
    # Maximum number of trials
    max_ntrial = 3
    # Minimum Error
    err_min = 10**(-14)
    # Which combinations of Refinement Form, Refinement Type, and Refinement Kind
    combos = [
        h_uni_ang
    ]
    
    # Test Output Parameters
    do_plot_mesh        = True
    do_plot_mesh_p      = True
    do_plot_uh          = True
    do_plot_u           = True
    do_plot_diff        = True
    do_plot_anl_err_ind = False
    do_plot_sol_vecs    = False
    do_calc_hi_res_err  = False
    do_calc_low_res_err = False
    do_plot_errs        = True

    # Which problems to solve
    prob_nums = []
    for x_num in range(2, 3):
        for y_num in range(2, 3):
            for th_num in range(0, 4):
                for scat_num in range(0, 3):
                    prob_nums += [[x_num, y_num, th_num, scat_num]]
                    
    for prob_num in [[2, 2, 3, 0]]:
        prob_dir = os.path.join(test_dir, str(prob_num))
        os.makedirs(prob_dir, exist_ok = True)
        
        msg = (
            'Starting problem {}...\n'.format(prob_num)
        )
        utils.print_msg(msg)
        
        combo_ndofs = {}
        combo_anl_errs = {}
        combo_hr_errs  = {}
        
        for combo in combos:
            combo_str = combo['short_name']
            combo_dir = os.path.join(prob_dir, combo_str)
            os.makedirs(combo_dir, exist_ok = True)
            
            msg = ( 'Starting combination {}...\n'.format(combo_str) )
            utils.print_msg(msg)
            
            # Get the base mesh, manufactured solution
            [Lx, Ly]                   = [2., 3.]
            pbcs                       = [False, False]
            mesh = ji_mesh.Mesh(Ls     = [Lx, Ly],
                                pbcs   = pbcs,
                                ndofs  = combo['ndofs'],
                                has_th = True)
            
            for _ in range(0, combo['nref_ang']):
                mesh.ref_mesh(kind = 'ang', form = 'h')
                
            for _ in range(0, combo['nref_spt']):
                mesh.ref_mesh(kind = 'spt', form = 'h')
            
            [u, kappa, sigma, Phi, f,
             u_intg_th, u_intg_xy] = get_cons_prob(prob_name = 'comp',
                                                   prob_num  = prob_num,
                                                   mesh      = mesh,
                                                   sth       = 64.)
            bcs_dirac = [u, [None, None, None]]
            
            if comm_rank == 0:
                kappa_file_name = 'kappa_{}.png'.format(prob_num[0:3])
                sigma_file_name = 'sigma_{}.png'.format(prob_num[0:3])
                Phi_file_name   = 'Phi_{}.png'.format(prob_num[3])
                gen_kappa_sigma_plots([Lx, Ly], kappa, sigma, prob_dir,
                                      [kappa_file_name, sigma_file_name])
                gen_Phi_plot(Phi, prob_dir, Phi_file_name)
                gen_u_plot([Lx, Ly], u, prob_dir)
                
            MPI_comm.Barrier()
            
            # Solve the manufactured problem over several trials
            ndofs = []
            errs  = []
            
            ndof  = mesh.get_ndof()
            trial = 0
            err   = 1.
            while ((ndof < max_ndof)
                   and (trial <= max_ntrial)
                   and (err > err_min)):
                ndof = mesh.get_ndof()
                ndofs += [ndof]
                
                perf_trial_0 = perf_counter()
                msg = (
                    '[Trial {}] Starting with '.format(trial) +
                    '{} of {} DoFs and '.format(ndof, max_ndof) +
                    'error {:.2E} of {:.2E}...\n'.format(err, err_min)
                )
                utils.print_msg(msg)
                
                # Set up output directories
                trial_dir = os.path.join(combo_dir, 'trial_{}'.format(trial))
                os.makedirs(trial_dir, exist_ok = True)
                
                uh_proj = get_soln(mesh, kappa, sigma, Phi, bcs_dirac, f,
                                   trial)
                err     = get_err(mesh, uh_proj, u, kappa, sigma, Phi,
                                  bcs_dirac, f,
                                  trial, trial_dir,
                                  nref_ang = combo['nref_ang'],
                                  nref_spt = combo['nref_spt'],
                                  ref_kind = combo['ref_kind'])
                errs += [err]
                
                if comm_rank == 0:
                    if do_plot_mesh:
                        gen_mesh_plot(mesh, trial, trial_dir)
                        
                    if do_plot_mesh_p:
                        gen_mesh_plot_p(mesh, trial, trial_dir)
                        
                    if do_plot_uh:
                        gen_uh_plot(mesh, uh_proj, trial, trial_dir)
                        
                if   combo['short_name'] == 'h-uni-ang':
                    mesh.ref_mesh(kind = 'ang', form = 'h')
                elif combo['short_name'] == 'p-uni-ang':
                    for _ in range(0, 3):
                        mesh.ref_mesh(kind = 'ang', form = 'p')
                        
                perf_trial_f    = perf_counter()
                perf_trial_diff = perf_trial_f - perf_trial_0
                msg = (
                    '[Trial {}] Trial completed! '.format(trial) +
                    'Time Elapsed: {:08.3f} [s]\n'.format(perf_trial_diff)
                )
                utils.print_msg(msg)
                
                trial += 1
                
            if comm_rank == 0:
                if do_plot_errs:
                    fig, ax = plt.subplots()
                    
                    colors = ['#000000', '#E69F00', '#56B4E9', '#009E73',
                              '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
                              '#882255']
                    
                    ax.scatter(ndofs, errs,
                               label = None,
                               color = colors[0])
                    
                    # Get best-fit line
                    [a, b] = np.polyfit(np.log10(ndofs), np.log10(errs), 1)
                    xx = np.logspace(np.log10(ndofs[0]), np.log10(ndofs[-1]))
                    yy = 10**b * xx**a
                    ax.plot(xx, yy,
                            label = '{} High-Res.: {:4.2f}'.format(combo_name, a),
                            color = colors[0],
                            linestyle = '--'
                            )
                    
                    ax.set_xscale('log', base = 10)
                    ax.set_yscale('log', base = 10)
                    
                    err_max = max(errs)
                    err_min = min(errs)
                    if np.log10(err_max) - np.log10(err_min) < 1:
                        ymin = 10**(np.floor(np.log10(err_min)))
                        ymax = 10**(np.ceil(np.log10(err_max)))
                        ax.set_ylim([ymin, ymax])
                        
                    ax.set_xlabel('Total Degrees of Freedom')
                    anl_err_str = (
                        r'$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u - u_{hp} \right)^2\,d\vec{x}\,d\hat{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\hat{s}}}$'
                    )
                    ax.set_ylabel(anl_err_str)
                    
                    ax.legend()
                    
                    title_str = ( '{} Convergence Rate'.format(combo['full_name']) )
                    ax.set_title(title_str)
                    
                    file_name = 'convergence.png'
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
                'Combination {} complete!\n'.format(combo['full_name']) +
                12 * ' ' + 'Time elapsed: {:08.3f} [s]\n'.format(perf_combo_dt)
            )
            utils.print_msg(msg)
            
        if comm_rank == 0:
            if do_plot_errs:
                fig, ax = plt.subplots()
                
                ncombo = len(combos)
                
                colors = ['#000000', '#E69F00', '#56B4E9', '#009E73',
                          '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
                          '#882255']
                
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
                            label = '{}: {:4.2f}'.format(combo_name, a),
                            color = colors[cc],
                            linestyle = '--'
                            )
                    
                ax.legend()
                
                ax.set_xscale('log', base = 10)
                ax.set_yscale('log', base = 10)
                
                ax.set_xlabel('Total Degrees of Freedom')
                anl_err_str = (
                    r'$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u - u_{hp} \right)^2\,d\vec{x}\,d\hat{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\hat{s}}}$'
                )
                ax.set_ylabel(anl_err_str)
                
                title_str = ( 'Convergence Rate' )
                ax.set_title(title_str)
                
                file_name = 'convergence.png'
                file_path = os.path.join(figs_dir, file_name)
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
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(r'$\kappa\left( x,\ y \right)$')
        
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
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(r'$\sigma\left( x,\ y \right)$')
        
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
        r_tick_labels = ['{:3.2f}'.format(r_tick) for r_tick in r_ticks]
        th_ticks = np.linspace(0, 2. * np.pi, num = 8, endpoint = False)
        th_tick_labels = [r'${:3.2f} \pi$'.format(th_tick/np.pi)
                          for th_tick in th_ticks]
        
        fig, ax = plt.subplots(subplot_kw = {'projection': 'polar'})
        
        Phi_plot = ax.plot(th, rr, color = 'black')
        
        ax.set_rlim([0, max_r])
        ax.set_rticks(r_ticks, r_tick_labels)
        ax.set_xlabel(r"$\theta - \theta'$")
        ax.set_xticks(th_ticks, th_tick_labels)
        ax.set_title(r"$\Phi\left( \theta - \theta' \right)$")
        
        plt.tight_layout()
        plt.savefig(file_path, dpi = 300)
        
        plt.close(fig)
    
def gen_u_plot(Ls, u, figs_dir):
    perf_0 = perf_counter()
    msg = ( 'Plotting analytic solution...\n'
           )
    utils.print_msg(msg)

    mesh = ji_mesh.Mesh(Ls     = Ls[:],
                        pbcs   = [False, False],
                        ndofs  = [8, 8, 8],
                        has_th = True)
    for _ in range(0, 4):
        mesh.ref_mesh(kind = 'ang', form = 'h')
    for _ in range(0, 4):
        mesh.ref_mesh(kind = 'spt', form = 'h')
        
    file_names = ['u_th.png', 'u_xy.png', 'u_xth.png', 'u_yth.png', 'u_xyth.png']
    file_paths = []
    is_file_paths = []
    for file_name in file_names:
        file_path     = os.path.join(figs_dir, file_name)
        file_paths   += [file_path]
        is_file_path += [os.path.isfile(file_path)]
        
    if not all(is_file_path):
        u_proj = proj.Projection(mesh, u)

    for file_path in file_paths:
        ## FIX THIS
        if not os.path.isfile(file_path):
            proj.utils.plot_th(mesh, u_proj, file_name = file_path)
            
        if not os.path.isfile(file_path):
            proj.utils.plot_xy(mesh, u_proj, file_name = file_path)
            
        if not os.path.isfile(file_path):
            proj.utils.plot_xth(mesh, u_proj, file_name = file_path)
            
        if not os.path.isfile(file_path):
            proj.utils.plot_yth(mesh, u_proj, file_name = file_path)
        
        if not os.path.isfile(file_path):
            proj.utils.plot_xyth(mesh, u_proj, file_name = file_path)
    
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = ( 'Analytic solution plotted!\n' +
            22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
           )
    utils.print_msg(msg)
    
    
def get_soln(mesh, kappa, sigma, Phi, bcs_dirac, f, trial):
    perf_0 = perf_counter()
    msg = (
        '[Trial {}] Obtaining numerical solution...\n'.format(trial)
    )
    utils.print_msg(msg)

    [uh_proj, info] = rt.rtdg(mesh, kappa, sigma, Phi, bcs_dirac, f,
                              verbose = True)
        
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = (
        '[Trial {}] Numerical solution obtained! : '.format(trial) +
        'Exit Code {} \n'.format(info) +
        22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
    )
    utils.print_msg(msg)

    return uh_proj

def get_err(mesh, uh_proj, u, kappa, sigma, Phi, bcs_dirac, f,
            trial, figs_dir, **kwargs):
    default_kwargs = {'ndof_x' : 8,
                      'ndof_y' : 8,
                      'ndof_th' : 16}
    kwargs = {**default_kwargs, **kwargs}
    
    perf_0 = perf_counter()
    msg = ( '[Trial {}] Obtaining error...\n'.format(trial)
           )
    utils.print_msg(msg)
    
    err = amr.total_anl_err(mesh, uh_proj, u)
    
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = (
        '[Trial {}] Error obtained! : {:.4E}\n'.format(trial, err) +
        22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
    )
    utils.print_msg(msg)
    
    return err

def gen_mesh_plot(mesh, trial, trial_dir):
    perf_0 = perf_counter()
    msg = ( '[Trial {}] Plotting mesh...\n'.format(trial)
           )
    utils.print_msg(msg)
    
    file_name = 'mesh_3d_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    ji_mesh.utils.plot_mesh(mesh      = mesh,
                            file_name = file_path,
                            plot_dim  = 3)
    
    file_name = 'mesh_2d_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    ji_mesh.utils.plot_mesh(mesh        = mesh,
                            file_name   = file_path,
                            plot_dim    = 2,
                            label_cells = (trial <= 2))
    
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = ( '[Trial {}] Mesh plotted!\n'.format(trial) +
            22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
           )
    utils.print_msg(msg)

def gen_mesh_plot_p(mesh, trial, trial_dir):
    perf_0 = perf_counter()
    msg = ( '[Trial {}] Plotting mesh polynomial degree...\n'.format(trial)
           )
    utils.print_msg(msg)
    
    file_name = 'mesh_3d_p_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    ji_mesh.utils.plot_mesh_p(mesh        = mesh,
                              file_name   = file_path,
                              plot_dim    = 3)
    
    file_name = 'mesh_2d_p_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    ji_mesh.utils.plot_mesh_p(mesh        = mesh,
                              file_name   = file_path,
                              plot_dim    = 2,
                              label_cells = (trial <= 3))
    
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = ( '[Trial {}] Mesh polynomial degree plotted!\n'.format(trial) +
            22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
           )
    utils.print_msg(msg)
    
def gen_uh_plot(mesh, uh_proj, trial, trial_dir):
    perf_0 = perf_counter()
    msg = (
        '[Trial {}] Plotting numerical solution...\n'.format(trial)
    )
    utils.print_msg(msg)
    
    file_name = 'uh_th_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    proj.utils.plot_th(mesh, uh_proj, file_name = file_path)
    
    file_name = 'uh_xy_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    proj.utils.plot_xy(mesh, uh_proj, file_name = file_path)
    
    file_name = 'uh_xth_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    proj.utils.plot_xth(mesh, uh_proj, file_name = file_path)
    
    file_name = 'uh_yth_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    proj.utils.plot_yth(mesh, uh_proj, file_name = file_path)
    
    file_name = 'uh_xyth_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    proj.utils.plot_xyth(mesh, uh_proj, file_name = file_path)
    
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = (
        '[Trial {}] Numerical solution plotted!\n'.format(trial) +
        22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
    )
    utils.print_msg(msg)
    
def gen_err_ind_plot(mesh, err_ind, trial, trial_dir, file_name):
    perf_0 = perf_counter()
    msg = (
        '[Trial {}] Plotting error indicators...\n'.format(trial)
    )
    utils.print_msg(msg)
    
    file_path = os.path.join(trial_dir, file_name)
    amr.utils.plot_error_indicator(mesh, err_ind, file_name = file_path)
    
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = (
        '[Trial {}] Error indicator plotted!\n'.format(trial) +
        22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
    )
    utils.print_msg(msg)
