import os, sys


sys.path.append('../src')
from dg.mesh import Mesh
from dg.mesh.utils import plot_mesh, plot_mesh_p, plot_nhbrs
from amr import rand_err, ref_by_ind

from utils import print_msg

def main(dir_name = 'figs'):
    """
    Generates and plots a random example mesh.
    """
    
    figs_dir = os.path.join(dir_name, 'mesh_figs')
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
        mesh.ref_mesh(kind = 'spt', form = 'h')
    for _ in range(0, 2):
        mesh.ref_mesh(kind = 'ang', form = 'h')
        
    mesh.ref_col(col_key = 2, kind = 'spt', form = 'h')
    mesh.ref_col(col_key = 11, kind = 'spt', form = 'h')
    mesh.ref_cell(col_key = 42, cell_key = 4, form = 'h')
    mesh.ref_cell(col_key = 42, cell_key = 9, form = 'h')
        
    file_name = 'mesh_3d_flat.png'
    file_path = os.path.join(figs_dir, file_name)
    plot_mesh(mesh      = mesh,
              file_name = file_path,
              plot_dim  = 3)
    
    file_name = 'mesh_3d_box.png'
    file_path = os.path.join(figs_dir, file_name)
    plot_mesh(mesh      = mesh,
              file_name = file_path,
              plot_dim  = 3,
              plot_style = 'box')
    
    file_name = 'mesh_2d.png'
    file_path = os.path.join(figs_dir, file_name)
    plot_mesh(mesh        = mesh,
              file_name   = file_path,
              plot_dim    = 2,
              label_cells = False)

    file_name = 'mesh_nhbrs.png'
    file_path = os.path.join(figs_dir, file_name)
    col_key = 12
    cell_key = 9
    plot_nhbrs(mesh, col_key, cell_key,
               file_name = file_path)
    
    file_name = 'mesh_3d_p_flat.png'
    file_path = os.path.join(figs_dir, file_name)
    plot_mesh_p(mesh        = mesh,
                file_name   = file_path,
                plot_dim    = 3)
    
    file_name = 'mesh_2d_p.png'
    file_path = os.path.join(figs_dir, file_name)
    plot_mesh_p(mesh        = mesh,
                file_name   = file_path,
                plot_dim    = 2,
                label_cells = False)
    
if __name__ == '__main__':
    main()
