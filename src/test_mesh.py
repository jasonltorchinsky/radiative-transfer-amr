from mesh import ji_mesh
from mesh.tools import plot_mesh, plot_col_nhbrs, plot_cell_nhbrs

import numpy as np
import sys, getopt
import os

def main(argv):

    dir_name = 'test_mesh'
    os.makedirs(dir_name, exist_ok = True)

    # Test 2D mesh
    Lx = 1.5
    Ly = 1.5
    dir_name = 'test_mesh'
    os.makedirs(dir_name, exist_ok = True)

    mesh = ji_mesh.Mesh(Ls = [1.75, 2], pbcs = [True, False])
    file_name = os.path.join(dir_name, 'mesh_0.png')
    plot_mesh(mesh, file_name = file_name, label_cells = True)

    for ii in range(0, 1):
        mesh.ref_mesh()
        file_name = os.path.join(dir_name, 'mesh_' + str(ii + 1) + '.png')
        plot_mesh(mesh, file_name = file_name, label_cells = True)

    mesh.ref_col(mesh.cols[1])
    file_name = os.path.join(dir_name, 'mesh_2.png')
    plot_mesh(mesh, file_name = file_name, label_cells = True)
        
    '''mesh.ref_col(mesh.cols[10])
    print(mesh)
    file_name = os.path.join(dir_name, 'mesh_3.png')
    plot_mesh(mesh, file_name = file_name, label_cells = True)'''

    for col in list(mesh.cols.values()):
        for ii in range(0, 2):
            col.ref_col()

    
    col = mesh.cols[4]
    cell = col.cells[3]
    col.ref_cell(cell)

    
    col = mesh.cols[9]
    cell = col.cells[3]
    col.ref_cell(cell)

    for col_key, col in sorted(mesh.cols.items()):
        #print(col)
        print('\n')

    col = mesh.cols[3]
    cell = col.cells[3]
        
    nhbr_cells = ji_mesh.get_cell_spt_nhbr(mesh, col = col, cell = cell,
                                           axis = 0, nhbr_loc = '-')

    file_name = os.path.join(dir_name, 'mesh_2_3d.png')
    plot_mesh(mesh, file_name = file_name, label_cells = False, plot_dim = 3)
        
    for cell in nhbr_cells:
        print(cell)
        print('\n')

    sys.exit(2)
    
    # Test the find neighbors function
    nhbr_locs = ['+', '-']
    for col in list(mesh.cols.values()):
        if col.is_lf:
            [i, j] = col.idx
            lv = col.lv
            
            file_name = 'nhbrs_{}_{}_{}.png'.format(i, j, lv)
            plot_col_nhbrs(mesh, col = col,
                           file_name = os.path.join(dir_name, file_name),
                           label_cells = False,
                           plot_dim = 2)

            for cell in list(col.cells.values()):
                if cell.is_lf:
                    cell_idx = cell.idx
                    cell_lv = cell.lv

                    file_name = 'nhbrs_{}_{}_{}_{}_{}.png'.format(i, j, lv,
                                                                  cell_idx,
                                                                  cell_lv)
                    plot_cell_nhbrs(mesh, col = col, cell = cell,
                                    file_name = os.path.join(dir_name, file_name),
                                    label_cells = False)
            
    

if __name__ == '__main__':

    main(sys.argv[1:])
