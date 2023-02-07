import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
import os, sys

sys.path.append('../../src')
import dg.quadrature as qd
from rad_amr import calc_scat_matrix, Projection_2D, push_forward

def sigma(x, y):

    return (x + 1)**6 * (np.sin(18 * np.pi * y) + 0.5)**2

def Phi(theta, phi):

    return (1.0 / (3.0 * np.pi)) \
        * (1 + (np.cos(theta) * np.cos(phi) + np.sin(theta) * np.sin(phi))**2)

def f(x, y, th):

    return (np.cos(200 * np.pi * y) + np.exp(5 * x)) * ((1/12) * (6 + np.cos(2 * th)))

def g(x, y, th):

    return f(x, y, th) / sigma(x, y)

def test_2(mesh, dir_name = 'test_rtdg'):
    """
    Creates a plots of the scattering matrix for a given mesh, sigma, and Phi.    
    """

    test_2_dir = os.path.join(dir_name, 'test_2')
    os.makedirs(test_2_dir, exist_ok = True)

    M_scat = calc_scat_matrix(mesh, sigma, Phi)

    ### VISUALIZE THE ENTIRE MATRIX
    fig, ax = plt.subplots()
    ax.spy(M_scat, marker = '.', markersize = 0.1, color = 'k')
    ax.set_title('Global Scattering Matrix')
    
    file_name = 'scat_matrix.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(test_2_dir, file_name), dpi = 300)
    plt.close(fig)
'''
    ### SOLVE SIMPLIFIED PROBLEM kappa * u = f
    ntrial = 7
    mesh_dAs = np.zeros([ntrial])
    L1_errors = np.zeros([ntrial])
    L2_errors = np.zeros([ntrial])
    for trial in range(0, ntrial):
        # Get number of degrees of spatial freedom of mesh
        for col_key, col in sorted(mesh.cols.items()):
            col_ndof = 0
            if col.is_lf:
                # Get column information
                [x0, y0, x1, y1] = col.pos
                [dx, dy] = [x1 - x0, y1 - y0]

                dA = dx * dy

                mesh_dAs[trial] = dA
        
        f_vec = get_forcing_vector(mesh, f)
        M_mass = calc_mass_matrix(mesh, kappa)
        apr_sol_vec = spsolve(M_mass, f_vec)
        anl_sol_vec = get_proj_vector(mesh, g)

        # Caluclate error
        L1_errors[trial] = mesh_dAs[trial] * np.sum(np.abs(anl_sol_vec - apr_sol_vec))
        L2_errors[trial] = mesh_dAs[trial] * np.sqrt(np.sum((anl_sol_vec - apr_sol_vec)**2))

        mesh.ref_mesh()

    # Plot approximated solution versus exact solution
    fig, ax = plt.subplots()

    print(mesh_dAs)
    print(L1_errors)
    print(L2_errors)
        
    ax.plot(mesh_dAs, L1_errors, label = 'L$^1$ Error',
            color = 'k', linestyle = '-')
    ax.plot(mesh_dAs, L2_errors, label = 'L$^2$ Error',
            color = 'r', linestyle = '-')

    ax.set_xscale('log', base = 2)
    ax.set_yscale('log', base = 10)
    
    ax.set_xlabel('Column Area ($dA = dx * dy$)')
    ax.set_ylabel('Error')
    
    ax.legend()
    ax.set_title('Uniform $h$-Refinement Convergence Rate')
    
    file_name = 'h-ref_acc.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(test_1_dir, file_name), dpi = 300)
    plt.close(fig)

def get_forcing_vector(mesh, f):
    """
    Create a global forcing vector corresponding to f.
    """
    
    # Create column indexing for constructing global forcing vector,
    # global solution vector
    col_idx = 0
    col_idxs = dict()
    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            col_idxs[col_key] = col_idx
            col_idx += 1

    ncols = col_idx # col_idx counts the number of existing columns in mesh
    f_col_vecs = [None] * ncols # Global vector is a 1-D vector
    
    # Unpack f into a column vectors
    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            # Get column information, quadrature weights
            col_idx = col_idxs[col_key]
            [x0, y0, x1, y1] = col.pos
            dx = x1 - x0
            dy = y1 - y0
            [ndof_x, ndof_y] = col.ndofs

            [xxb, w_x, yyb, w_y, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                      nnodes_y = ndof_y)

            xxf = push_forward(x0, x1, xxb)
            yyf = push_forward(y0, y1, yyb)
            
            # Create cell indexing for constructing column forcing vector
            cell_idx = 0
            cell_idxs = dict()
            for cell_key, cell in sorted(col.cells.items()):
                if cell.is_lf:
                    cell_idxs[cell_key] = cell_idx
                    cell_idx += 1

            ncells = cell_idx # cell_idx counts the number of existing cells in column
            f_cell_vecs = [None] * ncells # Column forcing vector is a 1-D vector
            
            for cell_key, cell in sorted(col.cells.items()):
                if cell.is_lf:
                    # Get cell information, quadrature weights
                    cell_idx   = cell_idxs[cell_key]
                    [th0, th1] = cell.pos
                    dth = th1 - th0
                    [ndof_th]  = cell.ndofs

                    [_, _, _, _, _, w_th] = qd.quad_xyth(nnodes_th = ndof_th)

                    def beta(ii, jj, aa):
                        val = ndof_th * ndof_y * ii \
                            + ndof_th * jj \
                            + aa
                        return val

                    dcoeff = dx * dy * dth / 8

                    # List of entries, values for constructing the cell mask
                    cell_ndof = ndof_x * ndof_y * ndof_th
                    f_cell_vec  = np.zeros([cell_ndof])
                    g_cell_vec  = np.zeros([cell_ndof])
                    for ii in range(0, ndof_x):
                        wx_i = w_x[ii]
                        for jj in range(0, ndof_y):
                            wy_j = w_y[jj]
                            f_ij = f(xxf[ii], yyf[jj])
                            for aa in range(0, ndof_th):
                                wth_a = w_th[aa]
                                
                                beta_idx = beta(ii, jj, aa)
                                
                                f_cell_vec[beta_idx] = dcoeff * wx_i * wy_j \
                                    * wth_a * f_ij

                    f_cell_vecs[cell_idx] = f_cell_vec

            f_col_vecs[col_idx] = np.concatenate(f_cell_vecs, axis = None)

    f_vec = np.concatenate(f_col_vecs, axis = None)

    return f_vec

def get_proj_vector(mesh, f):
    """
    Create a global forcing vector corresponding to f.
    """
    
    # Create column indexing for constructing global forcing vector,
    # global solution vector
    col_idx = 0
    col_idxs = dict()
    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            col_idxs[col_key] = col_idx
            col_idx += 1

    ncols = col_idx # col_idx counts the number of existing columns in mesh
    f_col_vecs = [None] * ncols # Global vector is a 1-D vector
    
    # Unpack f into a column vectors
    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            # Get column information, quadrature weights
            col_idx = col_idxs[col_key]
            [x0, y0, x1, y1] = col.pos
            dx = x1 - x0
            dy = y1 - y0
            [ndof_x, ndof_y] = col.ndofs

            [xxb, _, yyb, _, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                  nnodes_y = ndof_y)

            xxf = push_forward(x0, x1, xxb)
            yyf = push_forward(y0, y1, yyb)
            
            # Create cell indexing for constructing column forcing vector
            cell_idx = 0
            cell_idxs = dict()
            for cell_key, cell in sorted(col.cells.items()):
                if cell.is_lf:
                    cell_idxs[cell_key] = cell_idx
                    cell_idx += 1

            ncells = cell_idx # cell_idx counts the number of existing cells in column
            f_cell_vecs = [None] * ncells # Column forcing vector is a 1-D vector
            
            for cell_key, cell in sorted(col.cells.items()):
                if cell.is_lf:
                    # Get cell information, quadrature weights
                    cell_idx   = cell_idxs[cell_key]
                    [th0, th1] = cell.pos
                    dth = th1 - th0
                    [ndof_th]  = cell.ndofs

                    def beta(ii, jj, aa):
                        val = ndof_th * ndof_y * ii \
                            + ndof_th * jj \
                            + aa
                        return val


                    # List of entries, values for constructing the cell mask
                    cell_ndof = ndof_x * ndof_y * ndof_th
                    f_cell_vec  = np.zeros([cell_ndof])
                    g_cell_vec  = np.zeros([cell_ndof])
                    for ii in range(0, ndof_x):
                        for jj in range(0, ndof_y):
                            f_ij = f(xxf[ii], yyf[jj])
                            for aa in range(0, ndof_th):
                                
                                beta_idx = beta(ii, jj, aa)
                                
                                f_cell_vec[beta_idx] = f_ij

                    f_cell_vecs[cell_idx] = f_cell_vec

            f_col_vecs[col_idx] = np.concatenate(f_cell_vecs, axis = None)

    f_vec = np.concatenate(f_col_vecs, axis = None)

    return f_vec
'''
