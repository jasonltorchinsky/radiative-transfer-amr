import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import os, sys

from .gen_mesh import gen_mesh

sys.path.append('../../tests')
from test_cases import get_cons_prob

sys.path.append('../../src')
from rt import calc_mass_matrix, calc_scat_matrix, \
    calc_intr_conv_matrix, calc_bdry_conv_matrix
from amr import rand_err, ref_by_ind

from utils import print_msg

def test_0(dir_name = 'test_perf'):
    """
    Tests the performance of matrix contruction utilizing the contructed
    test problems with random refinement.
    """
    
    test_dir = os.path.join(dir_name, 'test_0')
    os.makedirs(test_dir, exist_ok = True)

    # Test parameters:
    # Problem Name: 'mass', 'scat'tering, 'conv'ection, 'comp'lete
    prob_name = ''
    # Problem Number
    prob_num  = None
    # Refinement Type: 'sin'gle column, 'uni'form, 'a'daptive 'm'esh 'r'efinement,
    # random ('rng')
    ref_type = ''
    # Refinement Kind: 's'pa't'ia'l', 'ang'ular, 'all'
    ref_kind = ''
    # Refinement Form: 'h', 'p'
    ref_form = ''
    # Refinement Tolerance
    tol_spt = 0.75
    tol_ang = 0.75
    # Maximum number of DOFs
    max_ndof = 2**14
    # Maximum number of trials
    max_ntrial = 32
    # Which combinations of Refinement Form, Refinement Type, and Refinement Kind
    combos = [
        ['h',  'rng', 'spt'],
        ['p',  'rng', 'spt'],
        ['h',  'rng', 'ang'],
        ['p',  'rng', 'ang']
    ]

    # Test Output Parameters
    do_plot_perf        = True
    
    prob_nums = []
    for x_num in range(0, 4):
        for y_num in range(0, 4):
            for th_num in range(0, 4):
                prob_nums += [[x_num, y_num, th_num]]

    for prob_num in [[2, 2, 2]]:
        prob_dir = os.path.join(test_dir, str(prob_num))
        os.makedirs(prob_dir, exist_ok = True)
        
        msg = ( 'Starting problem {}...\n'.format(prob_num) )
        print_msg(msg)
        
        for combo in combos:
            [ref_form, ref_type, ref_kind] = combo
            combo_str = '{}-{}-{}'.format(ref_form, ref_type, ref_kind)
            combo_dir = os.path.join(prob_dir, combo_str)
            os.makedirs(combo_dir, exist_ok = True)
            
            msg = ( 'Starting combination {}...\n'.format(combo_str) )
            print_msg(msg)
            
            # Get the base mesh, manufactured solution
            [Lx, Ly]                   = [2., 3.]
            pbcs                       = [False, False]
            [ndof_x, ndof_y, ndof_th]  = [3, 3, 3]
            has_th                     = True
            mesh = gen_mesh(Ls     = [Lx, Ly],
                            pbcs   = pbcs,
                            ndofs  = [ndof_x, ndof_y, ndof_th],
                            has_th = has_th)
            
            [u, kappa, sigma, Phi, _, _, _] = get_cons_prob(prob_name = 'comp',
                                                            prob_num  = prob_num,
                                                            mesh      = mesh)
            ndof = get_mesh_ndof(mesh)
            
            # Solve the manufactured problem over several trials
            ndofs  = []
            ncols  = []
            ncells = []
            mass_dts = []
            scat_dts = []
            intr_conv_dts = []
            bdry_conv_dts = []
            
            trial = 0
            while (ndof < max_ndof) and (trial < max_ntrial):
                # Set up output directories              
                ndofs  += [get_mesh_ndof(mesh)]
                ncols  += [get_mesh_ncol(mesh)]
                ncells += [get_mesh_ncell(mesh)]
                
                ## Mass matrix
                mass_t0 = perf_counter()
                M_mass  = calc_mass_matrix(mesh, kappa)
                mass_tf = perf_counter()
                mass_dt = mass_tf - mass_t0
                
                mass_dts += [mass_dt]
                
                ## Scattering matrix
                scat_t0 = perf_counter()
                M_scat  = calc_scat_matrix(mesh, sigma, Phi)
                scat_tf = perf_counter()
                scat_dt = scat_tf - scat_t0
                
                scat_dts += [scat_dt]
                
                ## Interior convection matrix
                intr_conv_t0 = perf_counter()
                M_intr_conv  = calc_intr_conv_matrix(mesh)
                intr_conv_tf = perf_counter()
                intr_conv_dt = intr_conv_tf - intr_conv_t0
                
                intr_conv_dts += [intr_conv_dt]
                
                ## Boundary convection matrix
                bdry_conv_t0 = perf_counter()
                M_bdry_conv  = calc_bdry_conv_matrix(mesh)
                bdry_conv_tf = perf_counter()
                bdry_conv_dt = bdry_conv_tf - bdry_conv_t0
                
                bdry_conv_dts += [bdry_conv_dt]
                
                
                # Refine the mesh for the next trial
                if ref_type == 'uni':
                    ## Refine the mesh uniformly
                    mesh.ref_mesh(kind = ref_kind, form = ref_form)
                elif ref_type == 'rng':
                    ## Refine the mesh randomly
                    rand_err_ind = rand_err(mesh, kind = ref_kind, form = ref_form)
                    
                    mesh = ref_by_ind(mesh, rand_err_ind,
                                      ref_ratio = tol_spt,
                                      form = ref_form)
                    
                trial += 1
                    
            if do_plot_perf:
                colors = ['#000000', '#E69F00', '#56B4E9', '#009E73',
                          '#F0E442', '#0072B2', '#D55E00', '#CC79A7',
                          '#882255']
                
                fig, ax = plt.subplots()
                
                ax.plot(ndofs, mass_dts,
                        label     = 'Mass',
                        color     = colors[0],
                        linestyle = '-')

                ax.plot(ndofs, scat_dts,
                        label     = 'Scattering',
                        color     = colors[1],
                        linestyle = '-')

                ax.plot(ndofs, intr_conv_dts,
                        label     = 'Interior Convection',
                        color     = colors[2],
                        linestyle = '-')

                ax.plot(ndofs, bdry_conv_dts,
                        label     = 'Boundary Convection',
                        color     = colors[3],
                        linestyle = '-')

                ax.legend()
                
                ax.set_xscale('log', base = 2)
                ax.set_yscale('log', base = 2)
                        
                ax.set_xlabel('Total Degrees of Freedom')
                ax.set_ylabel('Construction Time [s]')
                
                ref_strat_str = ''
                if ref_type == 'uni':
                    ref_strat_str = 'Uniform'
                elif ref_type == 'rng':
                    ref_strat_str = 'Random'
                    
                ref_kind_str = ''
                if ref_kind == 'spt':
                    ref_kind_str = 'Spatial'
                elif ref_kind == 'ang':
                    ref_kind_str = 'Angular'
                elif ref_kind == 'all':
                    ref_kind_str = 'Spatio-Angular'
                    
                title_str = ( '{} {} ${}$-Refinement '.format(ref_strat_str,
                                                              ref_kind_str,
                                                              ref_form) +
                              'Matrix Construction Time' )
                ax.set_title(title_str)
                    
                file_name = 'matrix_construction.png'
                file_path = os.path.join(combo_dir, file_name)
                fig.set_size_inches(6.5, 6.5)
                plt.savefig(file_path, dpi = 300)
                plt.close(fig)
                
                
def get_mesh_ndof(mesh):
    
    mesh_ndof = 0
    
    col_items = sorted(mesh.cols.items())
    
    for col_key, col in col_items:
        if col.is_lf:
            [nx, ny] = col.ndofs[:]
            
            cell_items = sorted(col.cells.items())
            
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    [nth] = cell.ndofs[:]
                    
                    cell_ndof = nx * ny * nth
                    
                    mesh_ndof += cell_ndof
                
    return mesh_ndof

def get_mesh_ncol(mesh):
    
    mesh_ncol = 0
    
    col_items = sorted(mesh.cols.items())
    
    for col_key, col in col_items:
        if col.is_lf:
            mesh_ncol += 1
                
    return mesh_ncol

def get_mesh_ncell(mesh):
    
    mesh_ncell = 0
    
    col_items = sorted(mesh.cols.items())
    
    for col_key, col in col_items:
        if col.is_lf:
            cell_items = sorted(col.cells.items())
            
            for cell_key, cell in cell_items:
                if cell.is_lf:
                    mesh_ncell += 1
                
    return mesh_ncell
