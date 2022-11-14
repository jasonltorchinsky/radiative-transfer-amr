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
                [dof_x, dof_y] = col.ndofs
                
                [nodes_x, _, nodes_y, _, _, _] = qd.quad_xya(dof_x, dof_y, 1)
                
                uh = np.zeros([dof_x, dof_y])
                for ii in range(0, dof_x):
                    x = x0 + (x1 - x0)/2 * (nodes_x[ii] + 1)
                    for jj in range(0, dof_y):
                        y = y0 + (y1 - y0)/2 * (nodes_y[jj] + 1)
                        uh[ii, jj] = u(x, y)

                self.cols[col_key] = uh


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
                # Create dict of cells for the column
                self.cols[col_key] = Projection_3D_Column()

                # Get information about the column
                [x0, y0, x1, y1] = col.pos
                [dof_x, dof_y] = col.ndofs
                
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

                        self.cols[col_key].cells[cell_key] = uh

class Projection_3D_Column:
    '''
    Collection of cells in each column.
    '''

    def __init__(self):
        self.cells = {}
    
                        
class Projection:
    '''
    Holds information about a function projected onto the nodes of the mesh.
    Note: This is stored as a dense matrix, although it may actually be sparse!
    '''

    def __init__(self, mesh, u = None, has_a = False):
        self.ndof = 0 # Total number of DOFs
        self.has_a = has_a # Whether or not there's an angular component

        # Put in a dummy zero fucntion i the function is not specified
        if not u:
            if has_a:
                def u(x, y, a):
                    return 0
            elif not has_a:
                def u(x, y):
                    return 0

        # If is has the angular component, we need it at each cell
        if has_a:
            for col_key, col in sorted(mesh.cols.items()):
                if col.is_lf:
                    [x0, y0, x1, y1] = col.pos
                    for cell_key, cell in sorted(col.cells.items()):
                        if cell.is_lf:
                            [a0, a1] = cell.pos
                            [dof_x, dof_y, dof_a] = cell.ndofs
                            
                            [nodes_x, _, nodes_y, _, nodes_a, _] = \
                                qd.quad_xya(dof_x, dof_y, dof_a)
                            
                            temp = np.zeros([dof_x, dof_y, dof_a])
                            for ii in range(0, dof_x):
                                x = x0 + (x1 - x0)/2 * (nodes_x[ii] + 1)
                                for jj in range(0, dof_y):
                                    y = y0 + (y1 - y0)/2 * (nodes_y[jj] + 1)
                                    for kk in range(0, dof_a):
                                        a = a0 + (a1 - a0)/2 * (nodes_a[kk] + 1)
                                        
                                        temp[ii, jj, kk] = u(x, y, a)
                                        
                            self.st_idxs[str([col_key, cell_key])] = self.ndof
                            self.ndof += dof_x * dof_y * dof_a
                            self.coeffs += list(temp.flatten(order = 'F'))
        else:
            for col_key, col in sorted(mesh.cols.items()):
                if col.is_lf:
                    [x0, y0, x1, y1] = col.pos
                    cell0 = sorted(col.cells.keys())[0]
                    cell = col.cells[cell0]
                    [dof_x, dof_y] = cell.ndofs[0:2]
                    
                    [nodes_x, _, nodes_y, _, _, _] = \
                        qd.quad_xya(dof_x, dof_y, 1)
                    
                    temp = np.zeros([dof_x, dof_y])
                    for ii in range(0, dof_x):
                        x = x0 + (x1 - x0)/2 * (nodes_x[ii] + 1)
                        for jj in range(0, dof_y):
                            y = y0 + (y1 - y0)/2 * (nodes_y[jj] + 1)
                            temp[ii, jj] = u(x, y)
                        
                    self.st_idxs[col_key] = self.ndof
                    self.ndof += dof_x * dof_y
                    self.coeffs += list(temp.flatten(order = 'F'))

    def __str__(self):
        msg = ( 'ndof:        {}\n'.format(self.ndof) +
                'has_a:       {}\n'.format(self.has_a) + 
                'st_idxs:     {}\n'.format(self.st_idxs)
               )

        return msg

    def __sub__(self, other):
        '''Difference of two projected solutions.'''
        if ( (self.ndof == other.ndof)
             and (self.has_a == other.has_a)
             and (self.st_idxs == other.st_idxs) ):
            new_proj = self
            new_proj.coeffs -= other.coeffs
        else:
            print('ERROR: Projections have mismatch, cannot subtract.')
            sys.exit(13)
            
        return new_proj

    '''def intg_a(self, mesh):
        #Integrates with respect to angle.
        if not self.has_a:
            print( ('WARNING: Attempted to integrate xy-projection ' +
                    + 'with respect to angle. Returning original projection.') )
            return self
        else:
            new_proj = Projection(mesh = mesh, has_a = False)
            for key, is_lf in sorted(mesh.is_lf.items()):
                if is_lf:
                    dof_x   = mesh.dof_x[key]
                    dof_y   = mesh.dof_y[key]
                    dof_a   = mesh.dof_a[key]
                    nelts_a = mesh.nelts_a[key]
                    
                    da = 2 * np.pi / nelts_a
                    
                    [_, _, _, _, _, weights_a] = qd.quad_xya(dof_x, dof_y, dof_a)

                    st_idx = self.st_idxs[key]
                    temp_elt = self.coeffs[st_idx:st_idx + dof_x * dof_y * nelts_a * dof_a]
                    temp_elt = np.array(temp_elt).reshape(dof_x * dof_y * nelts_a, dof_a, order = 'F')
                    temp_elt = temp_elt @ np.asarray(weights_a) * da / 2
                    temp_elt = temp_elt.reshape(dof_x, dof_y, nelts_a, order = 'F')
                    temp_elt = np.sum(temp_elt, axis = 2)

                    st_idx = new_proj.st_idxs[key]
                    new_proj.coeffs[st_idx:st_idx + dof_x * dof_y] \
                        = (temp_elt.flatten(order = 'F')).tolist()
                    
        return new_proj'''
        
