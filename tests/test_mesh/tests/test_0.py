import numpy as np
import matplotlib.pyplot as plt
import os, sys

sys.path.append('../../src')
from dg.mesh import ji_mesh, tools

def test_0(dir_name = 'test_mesh'):
    """
    Creates a simple 3-D mesh and visualizes it.
    """

    dir_name = os.path.join(dir_name, 'test_0')
    os.makedirs(dir_name, exist_ok = True)

    # Create the original 2-D mesh
    [Lx, Ly] = [3, 2]
    pbcs     = [True, False]

    mesh = ji_mesh.Mesh([Lx, Ly], pbcs, has_th = True)
    ncol_refs = 3
    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            for ref in range(0, ncol_refs):
                col.ref_col()
            
    
    # Refine the mesh some so we have a more interesting plot
    nrefs = 0
    fig, ax = plt.subplots()
    file_name = os.path.join(dir_name, 'mesh_2d_{}.png'.format(nrefs))
    tools.plot_mesh(mesh, ax = ax, file_name = file_name,
                    label_cells = True, plot_dim = 2)
    #file_name = os.path.join(dir_name, 'mesh_3d_{}.png'.format(nrefs))
    #tools.plot_mesh(mesh, ax = ax, file_name = file_name,
    #                label_cells = True, plot_dim = 3)
    
    
    nuni_ref = 2
    for ref in range(0, nuni_ref):
        mesh.ref_mesh()
        nrefs += 1
        fig, ax = plt.subplots()
        file_name = os.path.join(dir_name, 'mesh_2d_{}.png'.format(nrefs))
        tools.plot_mesh(mesh, ax = ax, file_name = file_name,
                    label_cells = True, plot_dim = 2)

        #fig, ax = plt.subplots()
        #file_name = os.path.join(dir_name, 'mesh_3d_{}.png'.format(nrefs))
        #tools.plot_mesh(mesh, ax = ax, file_name = file_name,
        #                label_cells = True, plot_dim = 3)

    ncol_ref = 5
    for ref in range(0, ncol_ref):
        col_keys = sorted(list(mesh.cols.keys()))
        mesh.ref_col(mesh.cols[col_keys[-1]])
        nrefs += 1
        fig, ax = plt.subplots()
        file_name = os.path.join(dir_name, 'mesh_2d_{}.png'.format(nrefs))
        tools.plot_mesh(mesh, ax = ax, file_name = file_name,
                        label_cells = False, plot_dim = 2)

        #fig, ax = plt.subplots()
        #file_name = os.path.join(dir_name, 'mesh_3d_{}.png'.format(nrefs))
        #tools.plot_mesh(mesh, ax = ax, file_name = file_name,
        #                label_cells = True, plot_dim = 3)

    file_name = os.path.join(dir_name, 'mesh_2d_bdry.png')
    tools.plot_mesh_bdry(mesh, file_name = file_name,
                         label_cells = False, plot_dim = 2)

    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            print('-- Column {}:'.format(col_key))
            for cell_key, cell in sorted(col.cells.items()):
                if cell.is_lf:
                    print(cell)


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

file_name = 'mesh_' + str(ref_num) + '_3d.png'
file_path = os.path.join(dir_name, file_name)
plot_mesh(mesh, file_name = file_path, plot_dim = 3)
print('Wrote (column refinement) {} mesh to {}\n'.format(ref_num, file_name))

for attr in ['dof_x', 'dof_y', 'lv']:
file_name = 'mesh_' + str(ref_num) + '_' + attr + '.png'
file_path = os.path.join(dir_name, file_name)
plot_mesh_attr(mesh, file_name = file_path, show_attr = attr)
print('Wrote {} mesh to {}\n'.format(attr, file_name))

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



quit()
'''
    
