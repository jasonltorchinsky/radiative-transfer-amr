from mesh import ji_mesh
from mesh.tools import plot_mesh, plot_col_nhbrs, plot_cell_nhbrs

import numpy as np
import sys, getopt
import os

def main(argv):

    print('Performing mesh refinement tests...\n')

    dir_name = 'test_mesh'
    os.makedirs(dir_name, exist_ok = True)

    # Test 2D mesh
    Lx = 2.0
    Ly = 1.5

    
    mesh = ji_mesh.Mesh(Ls = [Lx, Ly], pbcs = [True, False])
    file_name = 'mesh_0.png'
    file_path = os.path.join(dir_name, file_name)
    plot_mesh(mesh, file_name = file_path, label_cells = True)
    print('Wrote 0 mesh to {}\n'.format(file_name))


    ref_num = 0
    nunirefs = 1
    ncolrefs = 2
    
    for ii in range(0, nunirefs):
        mesh.ref_mesh()
        
        ref_num += 1
        file_name = 'mesh_' + str(ref_num) + '.png'
        file_path = os.path.join(dir_name, file_name)
        plot_mesh(mesh, file_name = file_path, label_cells = True)
        print('Wrote (uniform refinement) {} mesh to {}\n'.format(ref_num, file_name))
    

    # Something buggy here. refining column 4 should not refine any other column.
    # The error is in the key! [0, 0], 1 and [0, 1], 0 have the same key!
    for ii in range(0, ncolrefs):
        col_keys = sorted(list(mesh.cols.keys()))
        col = mesh.cols[col_keys[-1]]
        
        axes = [0, 1]
        nhbr_locs = ['+', '-']
        for axis in axes:
            for nhbr_loc in nhbr_locs:
                col_nhbrs = ji_mesh.get_col_nhbr(mesh, col = col,
                                                 axis = axis, nhbr_loc = nhbr_loc)
        
        mesh.ref_col(col)
        
        ref_num += 1
        file_name = 'mesh_' + str(ref_num) + '.png'
        file_path = os.path.join(dir_name, file_name)
        plot_mesh(mesh, file_name = file_path, label_cells = True)
        print('Wrote (column refinement) {} mesh to {}\n'.format(ref_num, file_name))
    
    
    # Test the find neighbors function
    for col in list(mesh.cols.values()):
        if col.is_lf:
            [i, j] = col.idx
            lv = col.lv
            
            file_name = 'nhbrs_{}_{}_{}.png'.format(i, j, lv)
            file_path = os.path.join(dir_name, file_name)
            plot_col_nhbrs(mesh, col = col,
                           file_name = file_path,
                           label_cells = False,
                           plot_dim = 2)
    
    for col in list(mesh.cols.values()):
        for ii in range(0, 2):
            col.ref_col()
    
    file_name = 'mesh_' + str(ref_num) + '_3d.png'
    file_path = os.path.join(dir_name, file_name)
    plot_mesh(mesh, file_name = file_path, plot_dim = 3)
    print('Wrote (column refinement) {} mesh to {}\n'.format(ref_num, file_name))
    
    quit()
    '''
    for col in list(mesh.cols.values()):
        for ii in range(0, 2):
            col.ref_col()

    
    col = mesh.cols[4]
    cell = col.cells[3]
    col.ref_cell(cell)

    
    col = mesh.cols[9]
    cell = col.cells[3]
    col.ref_cell(cell)

    #for col_key, col in sorted(mesh.cols.items()):
        #print(col)
        #print('\n')

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
    '''
    

if __name__ == '__main__':

    main(sys.argv[1:])
