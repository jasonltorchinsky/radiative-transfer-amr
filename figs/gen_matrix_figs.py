import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
import os, sys

sys.path.append("../src")
from dg.mesh import Mesh
from rt import calc_mass_matrix, calc_scat_matrix, \
    calc_intr_conv_matrix, calc_bdry_conv_matrix, \
    calc_forcing_vec

from amr import rand_err, ref_by_ind

from utils import print_msg

def main(kappa, sigma, Phi, dir_name = "figs"):
    """
    Generates extinction, scattering, interior convection, and boundary
    convection matrices.
    """
    
    figs_dir = os.path.join(dir_name, "matrix_figs")
    os.makedirs(figs_dir, exist_ok = True)
    
    # Get the base mesh
    [Lx, Ly]                   = [2., 2.]
    pbcs                       = [False, False]
    [ndof_x, ndof_y, ndof_th]  = [2, 2, 2]
    has_th                     = True
    mesh = Mesh(Ls     = [Lx, Ly],
                pbcs   = pbcs,
                ndofs  = [ndof_x, ndof_y, ndof_th],
                has_th = has_th)
    
    # Perform some uniform (angular or spatial) h-refinements to start
    for _ in range(0, 1):
        mesh.ref_mesh(kind = "spt", form = "h")
    for _ in range(0, 2):
        mesh.ref_mesh(kind = "ang", form = "h")
        
    # Randomly refine
    for _ in range(0, 5):
        rand_err_ind = rand_err(mesh, kind = "all", form = "h")
        
        mesh = ref_by_ind(mesh, rand_err_ind)
    
    M_mass = calc_mass_matrix(mesh, kappa)
    
    M_scat = calc_scat_matrix(mesh, sigma, Phi)
    
    M_intr_conv = calc_intr_conv_matrix(mesh)
    
    M_bdry_conv = calc_bdry_conv_matrix(mesh)
    
    global_matrices = {"Extinction" : M_mass,
                       "Scattering" : M_scat,
                       "Interior Propagation" : M_intr_conv,
                       "Boundary Propagation" : M_bdry_conv}
    
    # Plot a local-element, a local-column, and the global matrix
    col_keys = list(sorted(mesh.cols.keys()))
    col_key  = col_keys[-5]
    col      = mesh.cols[col_key]
    col_ndof = 0
    cell_items = sorted(col.cells.items())
    [nx, ny] = col.ndofs[:]
    for cell_key, cell in cell_items:
        if cell.is_lf:
            [cell_ndof] = cell.ndofs[:]
            col_ndof += cell_ndof
    col_ndof *= nx * ny
    
    cell_keys   = list(sorted(col.cells.keys()))
    cell_key    = cell_keys[-3]
    cell        = col.cells[cell_key]
    [cell_ndof] = cell.ndofs[:]
    cell_ndof *= nx * ny
    
    for matrix_name, global_matrix in global_matrices.items():
        dense_matrix = global_matrix
        
        # Plot local-element interaction matrix
        fig, ax = plt.subplots()
        
        ax.spy(dense_matrix[0:cell_ndof, 0:cell_ndof],
               marker     = "s",
               markersize = 4.,
               color      = "k")
                        
        title_str = "Local-Spatio-Angular Element {} Matrix".format(matrix_name)
        ax.set_title(title_str)
        
        matrix_fname = (matrix_name.replace(" ", "_")).lower()
        file_name = "spatio_angular_element_{}_matrix.png".format(matrix_fname)
        file_path = os.path.join(figs_dir, file_name)
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(file_path, dpi = 300, bbox_inches = "tight")
        plt.close(fig)
        
        # Plot local-column interaction matrix
        fig, ax = plt.subplots()
        
        ax.spy(dense_matrix[0:col_ndof, 0:col_ndof],
               marker     = "s",
               markersize = 2.,
               color      = "k")
                        
        title_str = "Local-Spatial Element {} Matrix".format(matrix_name)
        ax.set_title(title_str)
        
        matrix_fname = (matrix_name.replace(" ", "_")).lower()
        file_name = "spatial_element_{}_matrix.png".format(matrix_fname)
        file_path = os.path.join(figs_dir, file_name)
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(file_path, dpi = 300, bbox_inches = "tight")
        plt.close(fig)

        # Plot global matrix
        fig, ax = plt.subplots()
        
        ax.spy(dense_matrix,
               marker     = "s",
               markersize = .2,
               color      = "k")
                        
        title_str = "Global {} Matrix".format(matrix_name)
        ax.set_title(title_str)
        
        matrix_fname = (matrix_name.replace(" ", "_")).lower()
        file_name = "global_{}_matrix.png".format(matrix_fname)
        file_path = os.path.join(figs_dir, file_name)
        fig.set_size_inches(6.5, 6.5)
        plt.savefig(file_path, dpi = 300, bbox_inches = "tight")
        plt.close(fig)
        

def kappa(x, y):
    return np.exp(-((x - 1.)**2 + (y - 1.)**2))

def sigma(x, y):
    return 0.9 * kappa (x, y)

def Phi(th, phi):
    return (1 / (3. * np.pi)) * (1. + (np.cos(th - phi))**2)

if __name__ == "__main__":
    main(kappa, sigma, Phi)
