def get_idx_map(ndof_x, ndof_y, ndof_th):
    """
    Get the map from p,q,r => alpha, i,j,a => beta.
    NOTE: Keep ndof_x as an input in case the idx map changes in the future.
    """

    def idx_map(ii, jj, aa):
        idx = ndof_th * ndof_y * ii \
            + ndof_th * jj \
            + aa

        return idx

    return idx_map

def get_idx_inv_map(ndof_x, ndof_y, ndof_th):
    """
    Get the map from alpha => p,q,r, beta => i,j,a.
    NOTE: Keep ndof_x as an input in case the idx map changes in the future.
    """

    def idx_inv_map(idx):
        a = idx%ndof_th
        j = int((idx - a)/ndof_th)%ndof_y
        i = int((idx - (ndof_th * j + a)) / (ndof_th * ndof_y))

        return [i, j, a]

    return idx_inv_map


def get_col_idxs(mesh):
    """
    Column indices for constructing global vectors/matrices.
    """
    
    col_idx = 0
    col_idxs = dict()
    
    col_items = sorted(mesh.cols.items())
    
    for col_key, col in col_items:
        if col.is_lf:
            col_idxs[col_key] = col_idx
            col_idx += 1
            
    ncols = col_idx # col_idx counts the number of existing columns in mesh
    
    return [ncols, col_idxs]

def get_cell_idxs(mesh, col_key):
    """
    Cell indices for constructing column vectors/matrices.
    """

    col = mesh.cols[col_key]
    
    cell_idx = 0
    cell_idxs = dict()
    
    cell_items = sorted(col.cells.items())
    
    for cell_key, cell in cell_items:
        if cell.is_lf:
            cell_idxs[cell_key] = cell_idx
            cell_idx += 1
            
    ncells = cell_idx # cell_idx counts the number of existing cells in column
    
    return [ncells, cell_idxs]
