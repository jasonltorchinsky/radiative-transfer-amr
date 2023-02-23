import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append('../../src')
from rad_amr import get_intr_mask

def test_0(dir_name = 'test_rtdg'):
    """
    Creates a plots of the boundary mask to extract the interior and boundary
    entries of a matrix.
    """

    test_dir = os.path.join(dir_name, 'test_0')
    os.makedirs(test_dir, exist_ok = True)
    
    # Create the base mesh which will be refined in each trial.
    [Lx, Ly]                   = [3., 2.]
    [ndof_x, ndof_y, ndof_th]  = [2, 2, 2]
    mesh = ji_mesh.Mesh(Ls     = [Lx, Ly],
                        pbcs   = [False, False],
                        ndofs  = [ndof_x, ndof_y, ndof_th],
                        has_th = True)
    
    # Refine the mesh for initial trial
    for _ in range(0, 2):
        mesh.cols[0].ref_col()
    for _ in range(0, 1):
        mesh.ref_mesh()

    intr_mask = get_intr_mask(mesh)
    intr_mask_dense = intr_mask.astype(dtype = np.int32)
    
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
        
    ax.plot(intr_mask_dense, color = 'k', linestyle = '-',
            drawstyle = 'steps-post')
    ax.set_title('Interior Mask')
    
    file_name = 'intr_mask.png'
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(os.path.join(test_0_dir, file_name), dpi = 300)
    plt.close(fig)
