from .get_nhbr import get_col_nhbr

def get_bdry(mesh):

    nhbr_locs = ['+', '-']
    axes = [0, 1]

    ncols = 0
    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            ncols += 1

    bdrys = dict()
    bdrys['+0'] = np.zeros(ncols, dtype = np.bool_)
    bdrys['+1'] = np.zeros(ncols, dtype = np.bool_)
    bdrys['-0'] = np.zeros(ncols, dtype = np.bool_)
    bdrys['-1'] = np.zeros(ncols, dtype = np.bool_)
