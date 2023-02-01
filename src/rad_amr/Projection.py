import numpy as np
import sys

from dg import quadrature as qd

class Projection_2D:
    '''
    Projection of a function independent of angle.
    '''

    def __init__(self, mesh, u = None):
        self.cols = {}
        
        # Is no function is given, just use a zero
        if not u:
            def u(x, y):
                return 0

        # Calculate the projection in each column.
        for col_key, col in sorted(mesh.cols.items()):
            if col.is_lf:
                [x0, y0, x1, y1] = col.pos
                [ndof_x, ndof_y] = col.ndofs
                
                [nodes_x, _, nodes_y, _, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                              nnodes_y = ndof_y)
                xx = x0 + (x1 - x0)/2 * (nodes_x + 1)
                yy = y0 + (y1 - y0)/2 * (nodes_y + 1)
                
                uh = np.zeros([ndof_x, ndof_y])
                for ii in range(0, ndof_x):
                    for jj in range(0, ndof_y):
                        uh[ii, jj] = u(xx[ii], yy[ii])

                self.cols[col_key] = Column_2D([ndof_x, ndof_y], uh)

class Column_2D:

    def __init__(self, ndofs, vals):
        self.ndofs = ndofs
        self.vals  = vals

class Projection_3D:
    '''
    Projection of a function dependent on angle.

    Each column in the column dict needs to contains cells to match up with
    the Mesh class.
    '''

    def __init__(self, mesh, u = None):
        self.cols = {}
        
        # Is no function is given, just use a zero
        if not u:
            def u(x, y, a):
                return 0

        # Calculate the projection in each column.
        for col_key, col in sorted(mesh.cols.items()):
            if col.is_lf:
                # Get information about the column
                [x0, y0, x1, y1] = col.pos
                [dof_x, dof_y] = col.ndofs

                # Create dict of cells for the column
                self.cols[col_key] = Column_3D([dof_x, dof_y])
                
                [nodes_x, _, nodes_y, _, _, _] = qd.quad_xya(dof_x, dof_y, 1)

                x = x0 + (x1 - x0)/2 * (nodes_x + 1)
                y = y0 + (y1 - y0)/2 * (nodes_y + 1)

                for cell_key, cell in sorted(col.cells.items()):
                    if cell.is_lf:
                        [a0, a1] = cell.pos
                        [dof_a] = cell.ndofs

                        [_, _, _, _, nodes_a, _] = qd.quad_xya(1, 1, dof_a)
                        
                        uh = np.zeros([dof_x, dof_y, dof_a])
                        a = a0 + (a1 - a0)/2 * (nodes_a + 1)

                        for ii in range(0, dof_x):
                            for jj in range(0, dof_y):
                                for aa in range(0, dof_a):
                                    uh[ii, jj, aa] = u(x[ii], y[jj], a[aa])

                        self.cols[col_key].cells[cell_key] = Cell_3D(dof_a, uh)

class Column_3D:
    def __init__(self, ndofs):
        self.ndofs = ndofs
        self.cells = {}

class Cell_3D:

    def __init__(self, ndofs, vals):
        self.ndofs = ndofs
        self.vals  = vals
