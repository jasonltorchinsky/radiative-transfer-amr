def get_ndof(self):

    mesh_ndof = 0.
    for col_key, col in self.cols.items():
        if col.is_lf:
            [ndof_x, ndof_y] = col.ndofs[:]
            
            for cell_key, cell in col.cells.items():
                if cell.is_lf:
                    [ndof_th] = cell.ndofs[:]
                    
                    mesh_ndof += ndof_x * ndof_y * ndof_th
    return int(mesh_ndof)
