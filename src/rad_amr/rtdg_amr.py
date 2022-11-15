import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag
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
    fig = plt.figure(frameon = False)
    im_mass = plt.imshow(np.asarray(M_mass))
    plt.colobar()
    plt.savefig('M_mass.png', dpi = 300)
    plt.close(fig)
    
    #M_scat = calc_scat_matrix(mesh, sigma, phi)

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
                    
            col_mtxs[col_idx] = block_diag(cell_mtxs)
            col_idx += 1
                
    M_mass = block_diag(col_mtxs).asformat('csr')

    return M_mass

def calc_scat_matrix(mesh, sigma, phi):

    sigmah = Projection_2D(mesh, sigma)

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
            
            sigmah_col = sigmah.cols[col_key].vals
            
            # Store local-element matrices in here
            ncells = len(list(col.cells.keys()))**2
            cell_mtxs = ncells * [0]
            cell_idx = 0

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

                    for cell_key_1, cell_1 in sorted(col.cells.items()):
                        if cell_1.is_lf:
                            [dof_a_1] = cell_1.ndofs

                            # Indexing from i, j, a to beta
                            def beta(ii, jj, aa):
                                val = dof_a_1 * dof_y * ii \
                                    + dof_a * jj \
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
                                        
                            cell_mtxs[cell_idx] = coo_matrix((vlist, (alphalist, betalist)))
                            cell_idx += 1
                    
            col_mtxs[col_idx] = block_diag(cell_mtxs)
            col_idx += 1
                
    M_scat = block_diag(col_mtxs).asformat('csr')

    return M_scat

def calc_scat_matrix_orig(mesh, uh, sigma):

    '''ndof = uh.ndof
    sigmah = Projection(mesh = mesh, has_a = False, u = sigma)

    ilist = np.zeros([ndof], dtype = np.int32)
    jlist = np.zeros([ndof], dtype = np.int32)
    vlist = np.zeros([ndof])
    cnt = 0
    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos
            dx = x1 - x0
            dy = y1 - y0
            for cell_key, cell in sorted(col.cells.items()):
                if cell.is_lf:
                    [dof_x, dof_y, dof_a] = cell.ndofs
                    [a0, a1] = cell.pos
                    da = a1 - a0
                    
                    def local_elt_idx(ii, jj, aa):
                        val = dof_x * dof_y * aa\
                            + dof_x * jj \
                            + ii
                        return val

                    def pqr_to_ija(pp, qq, rr):
                        pp_pp = pp
                        

                    st_uh_idx = uh.st_idxs[str([col_key, cell_key])]
                    st_sigmah_idx = sigmah.st_idxs[col_key]
                    
                    temp_sigmah = sigmah.coeffs[st_sigmah_idx:st_sigmah_idx + dof_x * dof_y]
                    temp_sigmah = np.asarray(temp_sigmah).reshape(dof_x, dof_y, order = 'F')
        
                    [_, weights_x, _, weights_y, nodes_a, weights_a] = \
                        qd.quad_xya(dof_x, dof_y, dof_a)

                    for ii in range(0, dof_x):
                        for jj in range(0, dof_y):
                            for aa in range(0, dof_a):
                                for rr in range(0, dof_a):
                                    ilist[cnt] = st_uh_idx \
                                        + local_elt_idx(ii, jj, aa)
                                    jlist[cnt] = st_uh_idx \
                                        + idx(ii, jj, rr)
                                    vlist[cnt] = (da**2 / 4) \
                                        * weights_a[ll] * weights_a[ll_ll] \
                                        * f_scat( theta(kk_kk, nodes_a[ll_ll]),
                                                  theta(kk, nodes_a[ll]),
                                                  scat_type ) \
                                        * (dx * dy / 4) \
                                        * weights_x[ii] * weights_y[jj] \
                                        * temp_sigmah[ii, jj]
                                cnt += 1
                                
    M_scat = coo_matrix((vlist, (ilist, jlist))).asformat('csr')'''
