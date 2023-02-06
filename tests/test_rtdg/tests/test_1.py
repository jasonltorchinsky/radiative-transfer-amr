import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append('../../src')
from rad_amr import calc_mass_matrix

def test_1(mesh, kappa, dir_name = 'test_rtdg'):
    """
    Creates a plots of the mass matrix for a given mesh and kappa.
    """

    test_1_dir = os.path.join(dir_name, 'test_1')
    os.makedirs(test_1_dir, exist_ok = True)

    M_mass = calc_mass_matrix(mesh, kappa)

    # Plot of the entire matrix
    fig, ax = plt.subplots()
    ax.spy(M_mass, marker = '.', markersize = 0.1)
    ax.set_title('Global Mass Matrix')

    
    file_name = 'mass_matrix.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(test_1_dir, file_name), dpi = 300)
    plt.close(fig)

    # Plot of the main diagonal, with vertical gridlines denoting columns
    fig, ax = plt.subplots()

    mesh_ndof = 0
    # Plot a vertical line denoting where the column matrices are
    ax.axvline(x = mesh_ndof, color = 'gray', linestyle = '-',
               linewidth = 0.75)

    for col_key, col in sorted(mesh.cols.items()):
        col_ndof = 0
        if col.is_lf:
            [ndof_x, ndof_y] = col.ndofs
            for cell_key, cell in sorted(col.cells.items()):
                [ndof_th] = cell.ndofs

                cell_ndof = ndof_x * ndof_y * ndof_th

                col_ndof += cell_ndof

            mesh_ndof += col_ndof

            # Plot a vertical line denoting where the column matrices are
            ax.axvline(x = mesh_ndof, color = 'gray', linestyle = '-',
                       linewidth = 0.75)
        
    ax.plot(M_mass.diagonal(k = 0), color = 'k', linestyle = '-',
            drawstyle = 'steps-post')
    ax.set_title('Global Mass Matrix - Main Diagonal')
    
    file_name = 'mass_matrix_diag.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(test_1_dir, file_name), dpi = 300)
    plt.close(fig)
