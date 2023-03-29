def calc_col_key(idx, lv):
    ''' Get the key associated with each idxs, lvs. '''
    
    key = 0
    for ll in range(0, lv):
        key += 4 ** ll
        
    key += 2**lv * idx[1] + idx[0]
        
    return key

def calc_cell_key(idx, lv):

    key = 0
    for ll in range(0, lv):
        key += 2**ll

    key += idx

    return key
