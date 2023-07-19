import argparse
from mpi4py import MPI
import numpy as np
import petsc4py
from petsc4py import PETSc
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
from time import perf_counter
import os, sys


sys.path.append('../src')
from dg.mesh import Mesh
from dg.mesh.utils import plot_mesh, plot_mesh_p
from dg.matrix import get_intr_mask, split_matrix, merge_vectors
from dg.projection import Projection, to_projection
from dg.projection.utils import plot_xy, plot_xth, plot_yth, plot_xyth, plot_th
import dg.quadrature as qd
from rt import rtdg
from amr import cell_jump_err, col_jump_err, rand_err, high_res_err, \
    nneg_err, ref_by_ind, total_anl_err
from amr.utils import plot_error_indicator

import utils

def main():
    """
    Solves constructed ("manufactured") problems, with options for sub-problems
    and different types of refinement.
    """
    
    petsc4py.init()
    
    # MPI COMM for communicating data
    MPI_comm = MPI.COMM_WORLD
    
    #PETSc COMM for parallel matrix construction, solves
    comm = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()
    
    parser_desc = 'Determine which tests to run and where to put output.'
    parser = argparse.ArgumentParser(description = parser_desc)
    
    parser.add_argument('--dir', nargs = 1, default = 'figs',
                        required = False, help = 'Subdirectory to store output')
    help_str = 'Test Case Number - See Paper for Details'
    parser.add_argument('--test_num', nargs = 1, default = [1],
                        type = int, choices = [1, 2, 3, 4], required = False,
                        help = help_str)
    
    args = parser.parse_args()
    dir_name = args.dir
    test_num = args.test_num[0]
    
    figs_dir_name = 'test_{}_figs'.format(test_num)
    figs_dir = os.path.join(dir_name, figs_dir_name)
    if comm_rank == 0:
        os.makedirs(figs_dir, exist_ok = True)
    
    # Which combinations of Refinement Form, Refinement Type, and Refinement Kind
    combo_uangh = {'full_name'  : 'Uniform Angular h-Refinement',
                   'short_name' : 'h-uni-ang',
                   'ref_kind'   : 'ang'}
    combo_uangp = {'full_name'  : 'Uniform Angular p-Refinement',
                   'short_name' : 'p-uni-ang',
                   'ref_kind'   : 'ang'}
    combo_aangh = {'full_name'  : 'Adaptive Angular h-Refinement',
                   'short_name' : 'h-amr-ang',
                   'ref_kind'   : 'ang'}
    combo_aanghp = {'full_name'  : 'Adaptive Angular hp-Refinement',
                    'short_name' : 'hp-amr-ang',
                    'ref_kind'   : 'ang'}
    combo_aspthp = {'full_name'  : 'Adaptive Spatial hp-Refinement',
                    'short_name' : 'hp-amr-spt',
                    'ref_kind'   : 'spt'}
    combo_aallhp = {'full_name'  : 'Adaptive Spatio-Angular hp-Refinement',
                    'short_name' : 'hp-amr-all',
                    'ref_kind'    : 'all'}
    
    # Output options
    do_plot_mesh        = False
    do_plot_mesh_p      = True
    do_plot_uh          = True
    do_plot_err_ind     = True
    do_plot_errs        = True
    
    combo_names = []
    combo_ndofs = {}
    combo_errs  = {}
    
    perf_all_0 = perf_counter()
    msg = ( 'Generating test {} figures...\n'.format(test_num) )
    utils.print_msg(msg)

    # Parameters for mesh, and plot functions
    if test_num == 1:
        # End-Combo Parameters
        # Maximum number of DOFs
        max_ndof = 2**16
        # Maximum number of trials
        max_ntrial = 5
        # Minimum error before cut-off
        min_err = 3.e-6
        
        # Each combo in test has same starting mesh, but we give specifics here for flexibility
        [Lx, Ly] = [3., 2.]
        pbcs     = [False, False]
        ndofs    = [3, 3, 3]
        has_th   = True
        nref_ang = 3
        nref_spt = 2
        
        # Uniform Angular h-Refinement
        combo_uangh['Ls']       = [Lx, Ly]
        combo_uangh['pbcs']     = pbcs
        combo_uangh['ndofs']    = ndofs
        combo_uangh['has_th']   = has_th
        combo_uangh['nref_ang'] = nref_ang
        combo_uangh['nref_spt'] = nref_spt
        
        # Uniform Angular p-Refinement
        combo_uangp['Ls']       = [Lx, Ly]
        combo_uangp['pbcs']     = pbcs
        combo_uangp['ndofs']    = ndofs
        combo_uangp['has_th']   = has_th
        combo_uangp['nref_ang'] = nref_ang
        combo_uangp['nref_spt'] = nref_spt
        
        # Adaptive Angular h-Refinement
        combo_aangh['Ls']       = [Lx, Ly]
        combo_aangh['pbcs']     = pbcs
        combo_aangh['ndofs']    = ndofs
        combo_aangh['has_th']   = has_th
        combo_aangh['nref_ang'] = nref_ang
        combo_aangh['nref_spt'] = nref_spt
        
        # Adaptive Angular hp-Refinement
        combo_aanghp['Ls']       = [Lx, Ly]
        combo_aanghp['pbcs']     = pbcs
        combo_aanghp['ndofs']    = ndofs
        combo_aanghp['has_th']   = has_th
        combo_aanghp['nref_ang'] = nref_ang
        combo_aanghp['nref_spt'] = nref_spt
        
        [ndof_x_hr, ndof_y_hr, ndof_th_hr] = [None, None, None]
        combos = [
            combo_uangh,
            #combo_uangp,
            #combo_aangh,
            #combo_aanghp
        ]
        
    elif test_num == 2:
        # Each combo in test has same starting mesh, but we give specifics here fore flexibility
        [Lx, Ly] = [3., 2.]
        pbcs     = [True, False]
        has_th   = True
        
        # Uniform Angular h-Refinement
        combo_uangh['Ls']       = [Lx, Ly]
        combo_uangh['pbcs']     = pbcs
        combo_uangh['ndofs']    = [6, 6, 3]
        combo_uangh['has_th']   = has_th
        combo_uangh['nref_ang'] = 3
        combo_uangh['nref_spt'] = 2
        
        # Uniform Angular p-Refinement
        combo_uangp['Ls']       = [Lx, Ly]
        combo_uangp['pbcs']     = pbcs
        combo_uangp['ndofs']    = [4, 4, 3]
        combo_uangp['has_th']   = has_th
        combo_uangp['nref_ang'] = 3
        combo_uangp['nref_spt'] = 2
        
        # Adaptive Angular h-Refinement
        combo_aangh['Ls']       = [Lx, Ly]
        combo_aangh['pbcs']     = pbcs
        combo_aangh['ndofs']    = [4, 4, 3]
        combo_aangh['has_th']   = has_th
        combo_aangh['nref_ang'] = 3
        combo_aangh['nref_spt'] = 2
        
        # Adaptive Angular hp-Refinement
        combo_aanghp['Ls']       = [Lx, Ly]
        combo_aanghp['pbcs']     = pbcs
        combo_aanghp['ndofs']    = [4, 4, 3]
        combo_aanghp['has_th']   = has_th
        combo_aanghp['nref_ang'] = 3
        combo_aanghp['nref_spt'] = 2
        
        [ndof_x_hr, ndof_y_hr, ndof_th_hr] = [None, None, None]
        combos = [
            combo_uangh,
            combo_uangp,
            combo_aangh,
            combo_aanghp
        ]
        
    elif test_num == 3:
        # Each combo in each test has a different starting mesh.
        [Lx, Ly] = [3., 2.]
        pbcs     = [True, False]
        has_th   = True
        
        # Adaptive Spatial hp-Refinement
        combo_aspthp['Ls']       = [Lx, Ly]
        combo_aspthp['pbcs']     = pbcs
        combo_aspthp['ndofs']    = [3, 3, 16]
        combo_aspthp['has_th']   = has_th
        combo_aspthp['nref_ang'] = 3
        combo_aspthp['nref_spt'] = 2

        # Adaptive Angular hp-Refinement
        combo_aanghp['Ls']       = [Lx, Ly]
        combo_aanghp['pbcs']     = pbcs
        combo_aanghp['ndofs']    = [8, 8, 3]
        combo_aanghp['has_th']   = has_th
        combo_aanghp['nref_ang'] = 2
        combo_aanghp['nref_spt'] = 3
        
        # Adaptive Spatio-Angular hp-Refinement
        combo_aallhp['Ls']       = [Lx, Ly]
        combo_aallhp['pbcs']     = pbcs
        combo_aallhp['ndofs']    = [3, 3, 3]
        combo_aallhp['has_th']   = has_th
        combo_aallhp['nref_ang'] = 2
        combo_aallhp['nref_spt'] = 2
        
        [ndof_x_hr, ndof_y_hr, ndof_th_hr] = [8, 8, 16]
        combos = [
            combo_aspthp,
            combo_aanghp,
            combo_aallhp
        ]
        
    elif test_num == 4:
        # Each combo in each test has a different starting mesh.
        [Lx, Ly] = [3., 2.]
        pbcs     = [True, False]
        has_th   = True
        # Adaptive Spatial hp-Refinement
        combo_aspthp['Ls']       = [Lx, Ly]
        combo_aspthp['pbcs']     = pbcs
        combo_aspthp['ndofs']    = [3, 3, 16]
        combo_aspthp['has_th']   = has_th
        combo_aspthp['nref_ang'] = 3
        combo_aspthp['nref_spt'] = 2

        # Adaptive Angular hp-Refinement
        combo_aanghp['Ls']       = [Lx, Ly]
        combo_aanghp['pbcs']     = pbcs
        combo_aanghp['ndofs']    = [8, 8, 3]
        combo_aanghp['has_th']   = has_th
        combo_aanghp['nref_ang'] = 2
        combo_aanghp['nref_spt'] = 3
        
        # Adaptive Spatio-Angular hp-Refinement
        combo_aallhp['Ls']       = [Lx, Ly]
        combo_aallhp['pbcs']     = pbcs
        combo_aallhp['ndofs']    = [3, 3, 3]
        combo_aallhp['has_th']   = has_th
        combo_aallhp['nref_ang'] = 2
        combo_aallhp['nref_spt'] = 2
        
        [ndof_x_hr, ndof_y_hr, ndof_th_hr] = [8, 8, 16]
        combos = [
            combo_aspthp,
            combo_aanghp,
            combo_aallhp
        ]

    if test_num == 1:
        # Manufactured solution
        def X(x):
            return np.exp(-((1. / Lx) * (x - (Lx / 3.)))**2)
        def dXdx(x):
            return -(2. / Lx**2) * (x - (Lx / 3.)) * X(x)
        def Y(y):
            return np.exp(-4. * (Ly - y) / Ly)
        def dYdy(y):
            return (4. / Ly) * Y(y)
        def XY(x, y):
            return X(x) * Y(y)
        sth = 96.
        def Theta(th):
            return np.exp(-((sth / (2. * np.pi)) * (th - (7. * np.pi / 5.)))**2)
        def u(x, y, th):
            return XY(x, y) * Theta(th)
        
        def kappa_x(x):
            return np.exp(-((1. / Lx) * (x - (Lx / 2.)))**2)
        def kappa_y(y):
            return np.exp(-y / Ly)
        def kappa(x, y):
            return 10. * kappa_x(x) * kappa_y(y)
        
        def sigma(x, y):
            return 0.1 * kappa(x, y)
        
        def Phi(th, phi):
            val = (1. / (3. * np.pi)) * (1. + (np.cos(th - phi))**2)
            return val
        
        def f(x, y, th):
            # Propagation part
            prop = (np.cos(th) * dXdx(x) * Y(y) + np.sin(th) * X(x) * dYdy(y)) * Theta(th)
            # Extinction part
            extn = kappa(x, y) * u(x, y, th)
            # Scattering part
            [Theta_scat, _] = quad(lambda phi: Phi(th, phi) * Theta(phi), 0., 2. * np.pi,
                                   epsabs = 1.e-9, epsrel = 1.e-9, limit = 100, maxp1 = 100)
            scat =  sigma(x, y) * XY(x, y) * Theta_scat
            return prop + extn - scat
        
        def bcs(x, y, th):
            return u(x, y, th)
        dirac = [None, None, None]
        
        def u_intg_th(x, y, th0, th1):
            [Theta_intg, _] = quad(lambda th: Theta(th), th0, th1,
                                   epsabs = 1.e-9, epsrel = 1.e-9,
                                   limit = 100, maxp1 = 100)
            return XY(x, y) * Theta_intg
        
        def u_intg_xy(x0, x1, y0, y1, th):
            [XY_intg, _] = dblquad(lambda x, y: XY(x, y), x0, x1, y0, y1,
                                   epsabs = 1.e-9, epsrel = 1.e-9,
                                   limit = 100, maxp1 = 100)
            return XY_intg * Theta(th)
        
    elif test_num == 2:
        # Test problem : Clear sky with negligible scattering
        u = None
        
        def kappa_x(x):
            return np.ones_like(x)
        def kappa_y(y):
            return np.ones_like(y)
        def kappa(x, y):
            return 2.5 * kappa_x(x) * kappa_y(y)

        def sigma(x, y):
            return 0.9 * kappa(x, y)

        g = 0.925
        def Phi_HG(Th):
            return (1. - g**2) / (1 + g**2 - 2. * g * np.cos(Th))**(3./2.)
        [Phi_norm, abserr] = quad(lambda Th : Phi_HG(Th), 0., 2. * np.pi,
                                  epsabs = 1.e-9, epsrel = 1.e-9,
                                  limit = 100, maxp1 = 100)
        def Phi(th, phi):
            val = (1. - g**2) / (1 + g**2 - 2. * g * np.cos(th - phi))**(3./2.)
            return val / Phi_norm

        def f(x, y, th):
            return 0

        y_top = Ly
        def bcs(x, y, th):
            sth = 96.
            if (y == y_top):
                return np.exp(-((sth / (2. * np.pi)) * (th - (3. * np.pi / 2.)))**2)
            else:
                return 0
        dirac = [None, None, None]
        
    elif test_num == 3:
        # Test problem : Horizontally Homogeneous
        u = None

        def kappa_x(x):
            return np.ones_like(x)
        Ay = 0.5
        fy = 1. / Ly
        deltay = 0.5
        def kappa_y(y):
            return (2. * Ay / np.pi) * np.arctan(np.sin(2. * np.pi * fy * (y - Ly / 3.)) / deltay) + 0.5
        def kappa(x, y):
            return 9. * kappa_x(x) * kappa_y(y) + 0.1
        
        def sigma(x, y):
            return 0.9 * kappa(x, y)
        
        g = 0.925
        def Phi_HG(Th):
            return (1. - g**2) / (1 + g**2 - 2. * g * np.cos(Th))**(3./2.)
        [Phi_norm, abserr] = quad(lambda Th : Phi_HG(Th), 0., 2. * np.pi,
                                  epsabs = 1.e-9, epsrel = 1.e-9,
                                  limit = 100, maxp1 = 100)
        def Phi(th, phi):
            val = (1. - g**2) / (1 + g**2 - 2. * g * np.cos(th - phi))**(3./2.)
            return val / Phi_norm
        
        def f(x, y, th):
            return 0
        
        y_top = Ly
        def bcs(x, y, th):
            sth = 96.
            if (y == y_top):
                return np.exp(-((sth / (2. * np.pi)) * (th - (3. * np.pi / 2.)))**2)
            else:
                return 0
        dirac = [None, None, None]
        
    elif test_num == 4:
        # Test problem : Spatially inhomogeneous
        u = None

        Ax = 0.5
        fx = 1. / Lx
        deltax = 0.5
        def kappa_x(x):
            return (2. * Ax / np.pi) * np.arctan(np.sin(2. * np.pi * fx * (x - Lx / 3.)) / deltax) + 0.5
        Ay = 0.5
        fy = 1. / Ly
        deltay = 0.5
        def kappa_y(y):
            return (2. * Ay / np.pi) * np.arctan(np.sin(2. * np.pi * fy * (y - Ly / 3.)) / deltay) + 0.5
        def kappa(x, y):
            return 9. * kappa_x(x) * kappa_y(y) + 0.1

        def sigma(x, y):
            return 0.9 * kappa(x, y)

        g = 0.80
        def Phi_HG(Th):
            return (1. - g**2) / (1 + g**2 - 2. * g * np.cos(Th))**(3./2.)
        [Phi_norm, abserr] = quad(lambda Th : Phi_HG(Th), 0., 2. * np.pi,
                                  epsabs = 1.e-9, epsrel = 1.e-9,
                                  limit = 100, maxp1 = 100)
        def Phi(th, phi):
            val = (1. - g**2) / (1 + g**2 - 2. * g * np.cos(th - phi))**(3./2.)
            return val / Phi_norm

        def f(x, y, th):
            return 0
        y_top = Ly
        def bcs(x, y, th):
            sth = 96.
            if (y == y_top):
                return np.exp(-((sth / (2. * np.pi)) * (th - (3. * np.pi / 2.)))**2)
            else:
                return 0
        dirac = [None, None, None]
    
    # Plot extinction coefficient, scattering coefficient, and scattering phase function
    if comm_rank == 0 and False:
        kappa_file_name = 'kappa_{}.png'.format(test_num)
        sigma_file_name = 'sigma_{}.png'.format(test_num)
        Phi_file_name   = 'Phi_{}.png'.format(test_num)
        gen_kappa_sigma_plots([Lx, Ly], kappa, sigma, figs_dir,
                              [kappa_file_name, sigma_file_name])
        gen_Phi_plot(Phi, figs_dir, Phi_file_name)
        if test_num == 1:
            gen_u_plot([Lx, Ly], u, figs_dir)
    
    for combo in combos:
        combo_name = combo['short_name']
        combo_names += [combo_name]
        combo_dir = os.path.join(figs_dir, combo_name)
        if comm_rank == 0:
            os.makedirs(combo_dir, exist_ok = True)
        
        msg = ( 'Starting combination {}...\n'.format(combo['full_name']) )
        utils.print_msg(msg)
        
        perf_combo_0 = perf_counter()
        perf_setup_0 = perf_counter()
        
        # Get the base mesh
        mesh = Mesh(Ls     = combo['Ls'],
                    pbcs   = combo['pbcs'],
                    ndofs  = combo['ndofs'],
                    has_th = combo['has_th'])
        
        for _ in range(0, combo['nref_ang']):
            mesh.ref_mesh(kind = 'ang', form = 'h')
            
        for _ in range(0, combo['nref_spt']):
            mesh.ref_mesh(kind = 'spt', form = 'h')
        
        # Solve the problem over several trials
        ndofs = []
        errs  = []
        
        ndof      = mesh.get_ndof()
        prev_ndof = ndof
        trial = 0
        err   = 1.
        
        perf_setup_f = perf_counter()
        perf_setup_diff = perf_setup_f - perf_setup_0
        msg = ( 'Combination {} setup complete!\n'.format(combo['full_name']) +
                12 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_setup_diff)
               )
        utils.print_msg(msg)
        
        while (ndof < max_ndof) and (trial < max_ntrial):# and (err > min_err):
            ndof   = mesh.get_ndof()
            ndofs += [ndof]
            
            perf_trial_0 = perf_counter()
            msg = '[Trial {}] Starting with {} of {} ndofs and error {:.4E}...\n'.format(trial, ndof, max_ndof, err)
            utils.print_msg(msg)
            
            # Set up output directories
            trial_dir = os.path.join(combo_dir, 'trial_{}'.format(trial))
            if comm_rank == 0:
                os.makedirs(trial_dir, exist_ok = True)

            if test_num == 1:
                err_kind = 'anl'
            elif test_num == 2:
                err_kind = 'hr'
            elif test_num == 3:
                err_kind = 'hr'
            elif test_num == 4:
                err_kind = 'hr'
                
            uh_proj = get_soln(mesh, kappa, sigma, Phi, [bcs, dirac], f, trial)
            err     = get_err(mesh, uh_proj, u, kappa, sigma, Phi, [bcs, dirac], f,
                              trial, trial_dir, err_kind = err_kind,
                              ndof_x = ndof_x_hr,
                              ndof_y = ndof_y_hr,
                              ndof_th = ndof_th_hr,
                              nref_ang = combo['nref_ang'],
                              nref_spt = combo['nref_spt'],
                              ref_kind = combo['ref_kind'])
            errs   += [err]

            if comm_rank == 0 and False:
                if do_plot_mesh:
                    gen_mesh_plot(mesh, trial, trial_dir)
                    
                if do_plot_mesh_p:
                    gen_mesh_plot_p(mesh, trial, trial_dir)
                    
                if do_plot_uh:
                    gen_uh_plot(mesh, uh_proj, trial, trial_dir)
                    
            # Refine the mesh for the next trial
            if test_num == 1:
                neg_tol = -0.01
            elif test_num == 2:
                neg_tol = -0.01
            elif test_num == 3:
                neg_tol = -0.01
            elif test_num == 4:
                neg_tol = -0.01
                    
            if combo['short_name'] == 'h-uni-ang':
                mesh.ref_mesh(kind = 'ang', form = 'h')
            elif combo['short_name'] == 'p-uni-ang':
                mesh.ref_mesh(kind = 'ang', form = 'p')
            elif combo['short_name'] == 'h-amr-ang':
                uh_vec = uh_proj.to_vector()
                neg_tol = -0.01
                if np.any(uh_vec < neg_tol):
                    kwargs_ang = {'ref_col'      : False,
                                  'col_ref_form' : None,
                                  'col_ref_kind' : None,
                                  'col_ref_tol'  : None,
                                  'ref_cell'      : True,
                                  'cell_ref_form' : 'h',
                                  'cell_ref_kind' : 'ang',
                                  'cell_ref_tol'  : neg_tol}
                    err_ind_ang = nneg_err(mesh, uh_proj, **kwargs_ang)
                else:
                    kwargs_ang = {'ref_col'      : False,
                                  'col_ref_form' : None,
                                  'col_ref_kind' : None,
                                  'col_ref_tol'  : None,
                                  'ref_cell'      : True,
                                  'cell_ref_form' : 'h',
                                  'cell_ref_kind' : 'ang',
                                  'cell_ref_tol'  : 0.9}
                    
                    err_ind_ang = cell_jump_err(mesh, uh_proj, **kwargs_ang)
                
                if do_plot_err_ind:
                    gen_err_ind_plot(mesh, err_ind_ang, trial, trial_dir, 'err_ind_ang.png')
                    
                mesh = ref_by_ind(mesh, err_ind_ang)
                
            elif combo['short_name'] == 'hp-amr-ang':
                uh_vec = uh_proj.to_vector()
                neg_tol = -0.01
                if np.any(uh_vec < neg_tol):
                    kwargs_ang = {'ref_col'      : False,
                                  'col_ref_form' : None,
                                  'col_ref_kind' : None,
                                  'col_ref_tol'  : None,
                                  'ref_cell'      : True,
                                  'cell_ref_form' : 'h',
                                  'cell_ref_kind' : 'ang',
                                  'cell_ref_tol'  : neg_tol}
                    err_ind_ang = nneg_err(mesh, uh_proj, **kwargs_ang)
                else:
                    kwargs_ang = {'ref_col'      : False,
                                  'col_ref_form' : None,
                                  'col_ref_kind' : None,
                                  'col_ref_tol'  : None,
                                  'ref_cell'      : True,
                                  'cell_ref_form' : 'hp',
                                  'cell_ref_kind' : 'ang',
                                  'cell_ref_tol'  : 0.9}
                    
                    err_ind_ang = cell_jump_err(mesh, uh_proj, **kwargs_ang)
                
                if do_plot_err_ind:
                    gen_err_ind_plot(mesh, err_ind_ang, trial, trial_dir, 'err_ind_ang.png')
                    
                mesh = ref_by_ind(mesh, err_ind_ang)
                
            elif combo['short_name'] == 'hp-amr-spt':
                uh_vec = uh_proj.to_vector()
                neg_tol = -0.01
                if np.any(uh_vec < neg_tol):
                    kwargs_spt = {'ref_col'      : True,
                                  'col_ref_form' : 'h',
                                  'col_ref_kind' : 'spt',
                                  'col_ref_tol'  : neg_tol,
                                  'ref_cell'      : False,
                                  'cell_ref_form' : None,
                                  'cell_ref_kind' : None,
                                  'cell_ref_tol'  : None}
                    err_ind_spt = nneg_err(mesh, uh_proj, **kwargs_spt)
                else:
                    kwargs_spt = {'ref_col'      : True,
                                  'col_ref_form' : 'h',
                                  'col_ref_kind' : 'spt',
                                  'col_ref_tol'  : 0.9,
                                  'ref_cell'      : False,
                                  'cell_ref_form' : None,
                                  'cell_ref_kind' : None,
                                  'cell_ref_tol'  : None}
                    
                    err_ind_spt = col_jump_err(mesh, uh_proj, **kwargs_spt)
                
                if do_plot_err_ind:
                    gen_err_ind_plot(mesh, err_ind_spt, trial, trial_dir, 'err_ind_spt.png')
                    
                mesh = ref_by_ind(mesh, err_ind_spt)
                
            elif combo['short_name'] == 'hp-amr-all':
                uh_vec = uh_proj.to_vector()
                neg_tol = -0.01
                if np.any(uh_vec < neg_tol):
                    kwargs_spt = {'ref_col'      : True,
                                  'col_ref_form' : 'h',
                                  'col_ref_kind' : 'spt',
                                  'col_ref_tol'  : neg_tol,
                                  'ref_cell'      : False,
                                  'cell_ref_form' : None,
                                  'cell_ref_kind' : None,
                                  'cell_ref_tol'  : None}
                    err_ind_spt = nneg_err(mesh, uh_proj, **kwargs_spt)

                    kwargs_ang = {'ref_col'      : False,
                                  'col_ref_form' : None,
                                  'col_ref_kind' : None,
                                  'col_ref_tol'  : None,
                                  'ref_cell'      : True,
                                  'cell_ref_form' : 'h',
                                  'cell_ref_kind' : 'ang',
                                  'cell_ref_tol'  : neg_tol}
                    err_ind_ang = nneg_err(mesh, uh_proj, **kwargs_ang)
                else:
                    kwargs_spt = {'ref_col'      : True,
                                  'col_ref_form' : 'hp',
                                  'col_ref_kind' : 'spt',
                                  'col_ref_tol'  : 0.9,
                                  'ref_cell'      : False,
                                  'cell_ref_form' : None,
                                  'cell_ref_kind' : None,
                                  'cell_ref_tol'  : None}
                    
                    err_ind_spt = col_jump_err(mesh, uh_proj, **kwargs_spt)
                    
                    kwargs_ang = {'ref_col'      : False,
                                  'col_ref_form' : None,
                                  'col_ref_kind' : None,
                                  'col_ref_tol'  : None,
                                  'ref_cell'      : True,
                                  'cell_ref_form' : 'hp',
                                  'cell_ref_kind' : 'ang',
                                  'cell_ref_tol'  : 0.9}
                    
                    err_ind_ang = cell_jump_err(mesh, uh_proj, **kwargs_ang)
                
                if do_plot_err_ind:
                    gen_err_ind_plot(mesh, err_ind_spt, trial, trial_dir, 'err_ind_spt.png')
                    gen_err_ind_plot(mesh, err_ind_ang, trial, trial_dir, 'err_ind_ang.png')
                    
                mesh = ref_by_ind(mesh, err_ind_ang)
                mesh = ref_by_ind(mesh, err_ind_spt)
                
            perf_trial_f    = perf_counter()
            perf_trial_diff = perf_trial_f - perf_trial_0
            msg = (
                '[Trial {}] Trial completed!\n'.format(trial) +
                22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_trial_diff)
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
                if test_num == 1:
                    ax.set_ylabel(r'$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u - u_{hp} \right)^2\,d\vec{x}\,d\hat{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\hat{s}}}$')
                elif test_num == 2:
                    ax.set_ylabel(r'$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} - u_{hp} \right)^2\,d\vec{x}\,d\hat{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\hat{s}}}$')
                elif test_num == 3:
                    ax.set_ylabel(r'$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} - u_{hp} \right)^2\,d\vec{x}\,d\hat{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\hat{s}}}$')
                elif test_num == 3:
                    ax.set_ylabel(r'$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} - u_{hp} \right)^2\,d\vec{x}\,d\hat{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\hat{s}}}$')
            
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
            utils.print_msg(msg, blocking = False)
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
            if test_num == 1:
                ax.set_ylabel(r'$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u - u_{hp} \right)^2\,d\vec{x}\,d\hat{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\hat{s}}}$')
            elif test_num == 2:
                ax.set_ylabel(r'$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} - u_{hp} \right)^2\,d\vec{x}\,d\hat{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\hat{s}}}$')
            elif test_num == 3:
                ax.set_ylabel(r'$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} - u_{hp} \right)^2\,d\vec{x}\,d\hat{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\hat{s}}}$')
            elif test_num == 3:
                ax.set_ylabel(r'$\sqrt{\frac{\int_{\mathcal{S}} \int_{\Omega} \left( u_{hr} - u_{hp} \right)^2\,d\vec{x}\,d\hat{s}}{\int_{\mathcal{S}} \int_{\Omega} \left( u \right)^2\,d\vec{x}\,d\hat{s}}}$')
                
            title_str = ( 'Convergence Rate' )
            ax.set_title(title_str)
            
            file_name = 'convergence.png'
            file_path = os.path.join(figs_dir, file_name)
            fig.set_size_inches(6.5, 6.5)
            plt.tight_layout()
            plt.savefig(file_path, dpi = 300)
            plt.close(fig)
            
        perf_all_f = perf_counter()
        perf_all_dt = perf_all_f - perf_all_0
        msg = (
            'Test {} figures generated!\n'.format(test_num) +
            12 * ' ' + 'Time elapsed: {:08.3f} [s]\n'.format(perf_all_dt)
        )
        utils.print_msg(msg, blocking = False)

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
    fig, ax = plt.subplots()
    
    kappa_plot = ax.contourf(XX, YY, kappa_c, levels = 16, cmap = cmap, norm = norm)
    
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
    fig, ax = plt.subplots()
    
    kappa_plot = ax.contourf(XX, YY, sigma_c, levels = 16, cmap = cmap, norm = norm)
    
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(r'$\sigma\left( x,\ y \right)$')
    
    fig.colorbar(mpl.cm.ScalarMappable(norm = norm, cmap = cmap), ax = ax)
    
    file_path = os.path.join(figs_dir, file_names[1])
    plt.tight_layout()
    plt.savefig(file_path, dpi = 300)
    
    plt.close(fig)
    
def gen_Phi_plot(Phi, figs_dir, file_name):
    th = np.linspace(0, 2. * np.pi, num = 720)
    rr = Phi(0, th)
    
    max_r = max(rr)
    ntick = 2
    r_ticks = np.linspace(max_r / ntick, max_r, ntick)
    r_tick_labels = ['{:3.2f}'.format(r_tick) for r_tick in r_ticks]
    th_ticks = np.linspace(0, 2. * np.pi, num = 8, endpoint = False)
    th_tick_labels = [r'${:3.2f} \pi$'.format(th_tick/np.pi) for th_tick in th_ticks]
    
    fig, ax = plt.subplots(subplot_kw = {'projection': 'polar'})
    
    Phi_plot = ax.plot(th, rr, color = 'black')
    
    ax.set_rlim([0, max_r])
    ax.set_rticks(r_ticks, r_tick_labels)
    ax.set_xlabel(r"$\theta - \theta'$")
    ax.set_xticks(th_ticks, th_tick_labels)
    ax.set_title(r"$\Phi\left( \theta - \theta' \right)$")
    
    file_path = os.path.join(figs_dir, file_name)
    plt.tight_layout()
    plt.savefig(file_path, dpi = 300)
    
    plt.close(fig)
    
def gen_u_plot(Ls, u, figs_dir):
    perf_0 = perf_counter()
    msg = ( 'Plotting analytic solution...\n'
           )
    utils.print_msg(msg)

    mesh = Mesh(Ls     = Ls[:],
                pbcs   = [False, False],
                ndofs  = [8, 8, 8],
                has_th = True)
    for _ in range(0, 4):
        mesh.ref_mesh(kind = 'ang', form = 'h')
    for _ in range(0, 4):
        mesh.ref_mesh(kind = 'spt', form = 'h')

    u_proj = Projection(mesh, u)
    
    file_name = 'u_th.png'
    file_path = os.path.join(figs_dir, file_name)
    if not os.path.isfile(file_path):
        plot_th(mesh, u_proj, file_name = file_path)
    
    file_name = 'u_xy.png'
    file_path = os.path.join(figs_dir, file_name)
    if not os.path.isfile(file_path):
        plot_xy(mesh, u_proj, file_name = file_path)
    
    file_name = 'u_xth.png'
    file_path = os.path.join(figs_dir, file_name)
    if not os.path.isfile(file_path):
        plot_xth(mesh, u_proj, file_name = file_path)
    
    file_name = 'u_yth.png'
    file_path = os.path.join(figs_dir, file_name)
    if not os.path.isfile(file_path):
        plot_yth(mesh, u_proj, file_name = file_path)
    
    file_name = 'u_xyth.png'
    file_path = os.path.join(figs_dir, file_name)
    if not os.path.isfile(file_path):
        plot_xyth(mesh, u_proj, file_name = file_path)
    
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = ( 'Analytic solution plotted!\n' +
            22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
           )
    utils.print_msg(msg)
    
    
def get_soln(mesh, kappa, sigma, Phi, bcs_dirac, f, trial):
    perf_0 = perf_counter()
    msg = ( '[Trial {}] Obtaining numerical solution...\n'.format(trial)
           )
    utils.print_msg(msg)

    [uh_proj, info] = rtdg(mesh, kappa, sigma, Phi, bcs_dirac, f,
                           verbose = True)
        
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = ( '[Trial {}] Numerical solution obtained! : Exit Code {} \n'.format(trial, info) +
            22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
           )
    utils.print_msg(msg)

    return uh_proj

def get_err(mesh, uh_proj, u, kappa, sigma, Phi, bcs_dirac, f, trial, figs_dir, err_kind, **kwargs):
    default_kwargs = {'ndof_x' : 8,
                      'ndof_y' : 8,
                      'ndof_th' : 16}
    kwargs = {**default_kwargs, **kwargs}
    
    perf_0 = perf_counter()
    msg = ( '[Trial {}] Obtaining error...\n'.format(trial)
           )
    utils.print_msg(msg)

    if err_kind == 'anl':
        err = total_anl_err(mesh, uh_proj, u)
    elif err_kind == 'hr':
        # High-resolution error uses a different solver depending on the number of
        # DOFs in the high-resolution mesh.
        err = high_res_err(mesh, uh_proj,
                           kappa, sigma, Phi, bcs_dirac, f,
                           verbose = True,
                           dir_name = figs_dir,
                           **kwargs)
    
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = ( '[Trial {}] Error obtained! : {:.4E}\n'.format(trial, err) +
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
    plot_mesh(mesh      = mesh,
              file_name = file_path,
              plot_dim  = 3)
    
    file_name = 'mesh_2d_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    plot_mesh(mesh        = mesh,
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
    plot_mesh_p(mesh        = mesh,
                file_name   = file_path,
                plot_dim    = 3)
    
    file_name = 'mesh_2d_p_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    plot_mesh_p(mesh        = mesh,
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
    msg = ( '[Trial {}] Plotting numerical solution...\n'.format(trial)
           )
    utils.print_msg(msg)
    
    file_name = 'uh_th_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    plot_th(mesh, uh_proj, file_name = file_path)
    
    file_name = 'uh_xy_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    plot_xy(mesh, uh_proj, file_name = file_path)
    
    file_name = 'uh_xth_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    plot_xth(mesh, uh_proj, file_name = file_path)
    
    file_name = 'uh_yth_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    plot_yth(mesh, uh_proj, file_name = file_path)
    
    file_name = 'uh_xyth_{}.png'.format(trial)
    file_path = os.path.join(trial_dir, file_name)
    plot_xyth(mesh, uh_proj, file_name = file_path)
    
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = ( '[Trial {}] Numerical solution plotted!\n'.format(trial) +
            22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
           )
    utils.print_msg(msg)
    
def gen_err_ind_plot(mesh, err_ind, trial, trial_dir, file_name):
    perf_0 = perf_counter()
    msg = ( '[Trial {}] Plotting error indicators...\n'.format(trial)
           )
    utils.print_msg(msg)
    
    file_path = os.path.join(trial_dir, file_name)
    plot_error_indicator(mesh, err_ind, file_name = file_path)
    
    perf_f = perf_counter()
    perf_diff = perf_f - perf_0
    msg = ( '[Trial {}] Error indicator plotted!\n'.format(trial) +
            22 * ' ' + 'Time Elapsed: {:08.3f} [s]\n'.format(perf_diff)
           )
    utils.print_msg(msg)
    
if __name__ == '__main__':
    main()
