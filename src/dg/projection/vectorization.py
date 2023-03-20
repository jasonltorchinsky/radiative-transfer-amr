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
