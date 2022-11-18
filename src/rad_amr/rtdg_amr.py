import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag, bmat
from scipy.sparse.linalg import gmres
from scipy.linalg import eig
import sys
from time import perf_counter

from .Projection import Projection_2D

import matplotlib.pyplot as plt

import dg.quadrature as qd
from dg.mesh import ji_mesh, tools

import matplotlib.pyplot as plt

# Still need to test mass matrix and scattering matrix, at least to see if their forms are correct!

def rtdg_amr(mesh, uh_init, kappa, sigma, phi, **kwargs):
    '''
    Solve the radiative transfer problem.
    '''

    uh = uh_init

    # Construct the mass matrix
    #if kwargs['verbose']:
    #    print('Constructing the mass matrix...')
    #    t_start = perf_counter()
    
    M_mass = calc_mass_matrix(mesh, kappa)

    fig = plt.figure()
    im_mass = plt.spy(M_mass, marker = '.', markersize = 0.1)
    plt.savefig('M_mass.png', dpi = 500)
    plt.close(fig)

            
    M_scat = calc_scat_matrix(mesh, sigma, phi)

    fig = plt.figure()
    im_mass = plt.spy(M_scat, marker = '.', markersize = 0.1)
    plt.savefig('M_scat.png', dpi = 500)
    plt.close(fig)

    sys.exit(2)

def calc_mass_matrix(mesh, kappa):

    kappah = Projection_2D(mesh, kappa)

    # Store local-colmun matrices in here
    ncols = len(mesh.cols.keys())
    col_mtxs = ncols * [0]
    col_idx = 0

    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            # Get column information
            [x0, y0, x1, y1] = col.pos
            dx = x1 - x0
            dy = y1 - y0
            [dof_x, dof_y] = col.ndofs
            
            [_, weights_x, _, weights_y, _, _] = qd.quad_xya(dof_x, dof_y, 1)
            
            kappah_col = kappah.cols[col_key].vals
            
            # Store local-element matrices in here
            ncells = len(list(col.cells.keys()))
            cell_mtxs = ncells * [0]
            cell_idx = 0

            for cell_key, cell in sorted(col.cells.items()):
                if cell.is_lf:
                    [dof_a] = cell.ndofs

                    # Indexing from i, j, a to beta
                    # In this case, the alpha and beta indices are the same,
                    # so we don't have to do them separately
                    def beta(ii, jj, aa):
                        val = dof_a * dof_y * ii \
                            + dof_a * jj \
                            + aa
                        return val
                    
                    # Get cell information
                    cell_ndof = dof_x * dof_y * dof_a
                    [a0, a1] = cell.pos
                    da = a1 - a0
                    dcoeff = dx * dy * da / 8
                    
                    [_, _, _, _, _, weights_a] = qd.quad_xya(1, 1, dof_a)

                    # Lists for constructing diagonal matrices
                    cell_mtx_ndof = dof_x * dof_y * dof_a
                    alphalist = np.zeros([cell_mtx_ndof],
                                         dtype = np.int32) # alpha index
                    betalist = np.zeros([cell_mtx_ndof],
                                        dtype = np.int32) # beta index
                    vlist = np.zeros([cell_mtx_ndof]) # Matrix entry
                    cnt = 0
                    
                    for ii in range(0, dof_x):
                        wx_i = weights_x[ii]
                        for jj in range(0, dof_y):
                            wy_j = weights_y[jj]
                            kappa_ij = kappah_col[ii, jj]
                            for aa in range(0, dof_a):
                                wth_a = weights_a[aa]

                                # specify that the entry is on
                                betalist[cnt] = beta(ii, jj, aa)
                                alphalist[cnt] = betalist[cnt]
                                
                                vlist[cnt] = dcoeff * wx_i * wy_j * wth_a * kappa_ij
                                cnt += 1
                                
                    cell_mtxs[cell_idx] = coo_matrix((vlist, (alphalist, betalist)))
                    cell_idx += 1
                    
            col_mtxs[col_idx] = block_diag(cell_mtxs, format = 'csr')
            
            col_idx += 1
                
    M_mass = block_diag(col_mtxs, format = 'csr')

    return M_mass

def calc_scat_matrix(mesh, sigma, phi):

    sigmah = Projection_2D(mesh, sigma)

    # Store local-colmun matrices in here
    ncols = len(mesh.cols.keys())
    col_mtxs = ncols * [0]
    col_idx = 0 # I don't think I actually increment this, which is certainly a bug!

    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            # Get column information
            [x0, y0, x1, y1] = col.pos
            dx = x1 - x0
            dy = y1 - y0
            [dof_x, dof_y] = col.ndofs
            
            [_, weights_x, _, weights_y, _, _] = qd.quad_xya(dof_x, dof_y, 1)
            
            sigmah_col = sigmah.cols[col_key].vals
            
            # Store local-element matrices in here
            ncells = len(list(col.cells.keys()))
            cell_mtxs = ncells * [ncells * [0]]

            # _0 refers to element K' in the equations
            # _1 refers to element K in the equations
            cell_idx_0 = 0
            for cell_key_0, cell_0 in sorted(col.cells.items()):
                if cell_0.is_lf:
                    [dof_a_0] = cell_0.ndofs

                    # Indexing from i, j, a to beta
                    def alpha(pp, qq, rr):
                        val = dof_a_0 * dof_y * pp \
                            + dof_a_0 * qq \
                            + rr
                        return val
                    
                    # Get cell information
                    cell_0_ndof = dof_x * dof_y * dof_a_0
                    [a0_0, a1_0] = cell_0.pos
                    da_0 = a1_0 - a0_0
                    dcoeff = dx * dy * da_0 / 8
                    
                    [_, _, _, _, nodes_a_0, weights_a_0] = qd.quad_xya(1, 1, dof_a_0)
                    th_0 = a0_0 + (a1_0 - a0_0) / 2 * (nodes_a_0 + 1)

                    cell_idx_1 = 0
                    for cell_key_1, cell_1 in sorted(col.cells.items()):
                        if cell_1.is_lf:
                            [dof_a_1] = cell_1.ndofs

                            # Indexing from i, j, a to beta
                            def beta(ii, jj, aa):
                                val = dof_a_1 * dof_y * ii \
                                    + dof_a_1 * jj \
                                    + aa
                                return val

                            # Get cell information
                            cell_1_ndof = dof_x * dof_y * dof_a_1
                            [a0_1, a1_1] = cell_1.pos
                            da_1 = a1_1 - a0_1

                            [_, _, _, _, nodes_a_1, weights_a_1] = qd.quad_xya(1, 1, dof_a_1)
                            th_1 = a0_1 + (a1_1 - a0_1) / 2 * (nodes_a_1 + 1)

                            # Lists for constructing diagonal matrices
                            cell_mtx_ndof = dof_a_0 * dof_a_1 \
                                * dof_x * dof_y
                            alphalist = np.zeros([cell_mtx_ndof],
                                                 dtype = np.int32) # alpha index
                            betalist = np.zeros([cell_mtx_ndof],
                                                dtype = np.int32) # beta index
                            vlist = np.zeros([cell_mtx_ndof]) # Matrix entry
                            cnt = 0
                            
                            for ii in range(0, dof_x):
                                wx_i = weights_x[ii]
                                for jj in range(0, dof_y):
                                    wy_j = weights_y[jj]
                                    sigma_ij = sigmah_col[ii, jj]
                                    for rr in range(0, dof_a_0):
                                        wth_a_0 = weights_a_0[rr]
                                        for aa in range(0, dof_a_1):
                                            wth_a_1 = weights_a_1[aa]

                                            phi_ar = phi(th_0[rr], th_1[aa])

                                            # Index of entry
                                            alphalist[cnt] = alpha(ii, jj, aa)
                                            betalist[cnt] = beta(ii, jj, rr)
                                            
                                            vlist[cnt] = dcoeff * (da_1 / 2.0) * wx_i * wy_j * wth_a_0 * wth_a_1 * sigma_ij * phi_ar
                                            cnt += 1
                                        
                            cell_mtxs[cell_idx_0][cell_idx_1] = coo_matrix((vlist, (alphalist, betalist)))
                            cell_idx_1 += 1
                    cell_idx_0 += 1
                    
            col_mtxs[col_idx] = bmat(cell_mtxs, format = 'csr')
            
            col_idx += 1
                
    M_scat = block_diag(col_mtxs, format = 'csr')

    return M_scat

def calc_int_scat_matrix(mesh):

    # Store local-colmun matrices in here
    ncols = len(mesh.cols.keys())
    col_mtxs = ncols * [0]
    col_idx = 0

    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            # Get column information
            [x0, y0, x1, y1] = col.pos
            dx = x1 - x0
            dy = y1 - y0
            [dof_x, dof_y] = col.ndofs
            
            [nodes_x, weights_x, nodes_y, weights_y, _, _] = qd.quad_xya(dof_x, dof_y, 1)
            
            # Store local-element matrices in here
            ncells = len(list(col.cells.keys()))
            cell_mtxs = ncells * [0]
            cell_idx = 0

            for cell_key, cell in sorted(col.cells.items()):
                if cell.is_lf:
                    [dof_a] = cell.ndofs

                    # Indexing from i, j, a to beta
                    def beta(ii, jj, aa):
                        val = dof_a * dof_y * ii \
                            + dof_a * jj \
                            + aa
                        return val
                    
                    # Get cell information
                    cell_ndof = dof_x * dof_y * dof_a
                    [a0, a1] = cell.pos
                    da = a1 - a0
                    dcoeff = dx * dy * da / 8
                    
                    [_, _, _, _, nodes_a, weights_a] = qd.quad_xya(1, 1, dof_a)
                    th = a0 + (a1 - a0) / 2 * (nodes_a + 1)

                    # Construct delta_ip * delta_ar term
                    cell_mtx_ndof_ipar = dof_x * dof_y**2 * dof_a # Number of non-zero terms
                    
                    alphalist_ipar = np.zeros([cell_mtx_ndof_ipar],
                                         dtype = np.int32) # alpha index
                    betalist_ipar = np.zeros([cell_mtx_ndof_ipar],
                                        dtype = np.int32) # beta index
                    vlist_ipar = np.zeros([cell_mtx_ndof_ipar]) # Matrix entry
                    cnt = 0

                    for ii in range(0, dof_x):
                        wx_i = weights_x[ii]
                        for jj in range(0, dof_y):
                            wy_j = weights_y[jj]
                            for aa in range(0, dof_a):
                                wth_a = weights_a[aa]
                                for qq in range(0, dof_y):
                                    betalist_ipar[cnt] = beta(ii, qq, aa)
                                    alphalist_ipar[cnt] = beta(ii, jj, aa)
                                    # NEED TO FIX THIS FORMULA
                                    vlist_ipar[cnt] = dcoeff * wx_i * wy_j * wth_a \
                                        * qd.gl_deriv(nodes_y, qq, nodes_y[jj]) * np.sin(th[aa])
                                    cnt += 1

                    # Construct delta_jq * delta_ar term
                    cell_mtx_ndof_jqar = dof_x**2 * dof_y dof_a # Number of non-zero terms
                    
                    alphalist_jqar = np.zeros([cell_mtx_ndof_ipar],
                                         dtype = np.int32) # alpha index
                    betalist_jqar = np.zeros([cell_mtx_ndof_ipar],
                                        dtype = np.int32) # beta index
                    vlist_jqar = np.zeros([cell_mtx_ndof_ipar]) # Matrix entry
                    cnt = 0
                    
                    for ii in range(0, dof_x):
                        wx_i = weights_x[ii]
                        for jj in range(0, dof_y):
                            wy_j = weights_y[jj]
                            for aa in range(0, dof_a):
                                wth_a = weights_a[aa]
                                for pp in range(0, dof_x):
                                    betalist_jqar[cnt] = beta(pp, jj, aa)
                                    alphalist_jqar[cnt] = beta(ii, jj, aa)
                                    # NEED TO FIX THIS FORMULA
                                    vlist_jqar[cnt] = dcoeff * wx_i * wy_j * wth_a \
                                        * qd.gl_deriv(nodes_x, pp, nodes_x[ii]) * np.cos(th[aa])
                                    cnt += 1
                                
                    cell_mtxs[cell_idx] = coo_matrix((vlist_ipar, (alphalist_ipar, betalist_ipar))) + coo_matrix((vlist_jqar, (alphalist_jqar, betalist_jqar)))
                    cell_idx += 1
                    
            col_mtxs[col_idx] = block_diag(cell_mtxs, format = 'csr')
            
            col_idx += 1
                
    M_int_conv = block_diag(col_mtxs, format = 'csr')

    return M_int_conv
