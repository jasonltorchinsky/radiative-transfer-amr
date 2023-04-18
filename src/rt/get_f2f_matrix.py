import dg.projection as proj

def get_f2f_matrix(mesh, col_key_0, cell_key_0,
                   col_key_1, cell_key_1, F):
    '''
    Parse inputs here to the keys in the f2f_matrices dict, then
    get that matrix.
    '''
    
    col_0 = mesh.cols[col_key_0]
    col_1 = mesh.cols[col_key_1]

    if cell_key_0 is None and cell_key_1 is None:
        # Dealing with columns
        if F%2 == 0:
            dim_str = 'y'
            
            ndof_0 = col_0.ndofs[1]
            lv_0   = col_0.lv
            
            ndof_1 = col_1.ndofs[1]
            lv_1   = col_1.lv
            
            if lv_1 - lv_0 == 0:
                pos_str = 's'
            elif lv_1 - lv_0 == 1:
                if F == 0:
                    l_idx = 0
                    u_idx = 1
                else: # F == 2
                    l_idx = 1
                    u_idx = 0
                
                # Assume here that col_0, col_1 are indeed neighbors
                if col_0.nhbr_keys[F][l_idx] == col_key_1:
                    pos_str = 'l'
                elif col_0.nhbr_keys[F][u_idx] == col_key_1:
                    pos_str = 'u'
            elif lv_1 - lv_0 == -1:
                if F == 0:
                    l_idx = 1
                    u_idx = 0
                else: # F == 2
                    l_idx = 0
                    u_idx = 1
                
                # Assume here that col_0, col_1 are indeed neighbors
                if col_1.nhbr_keys[(F+2)%4][l_idx] == col_key_1:
                    pos_str = 'l'
                elif col_0.nhbr_keys[(F+2)%4][u_idx] == col_key_1:
                    pos_str = 'u'
                    
                    
        elif F%2 == 1:
            dim_str = 'x'
            
            ndof_0 = col_0.ndofs[0]
            lv_0   = col_0.lv
            
            ndof_1 = col_1.ndofs[0]
            lv_1   = col_1.lv

            if lv_1 - lv_0 == 0:
                pos_str = 's'
            elif lv_1 - lv_0 == 1:
                if F == 1:
                    l_idx = 1
                    u_idx = 0
                else: # F == 3
                    l_idx = 0
                    u_idx = 1
                
                # Assume here that col_0, col_1 are indeed neighbors
                if col_0.nhbr_keys[F][l_idx] == col_key_1:
                    pos_str = 'l'
                elif col_0.nhbr_keys[F][u_idx] == col_key_1:
                    pos_str = 'u'
            elif lv_1 - lv_0 == -1:
                if F == 1:
                    l_idx = 0
                    u_idx = 1
                else: # F == 3
                    l_idx = 1
                    u_idx = 0
                
                # Assume here that col_0, col_1 are indeed neighbors
                if col_1.nhbr_keys[(F+2)%4][l_idx] == col_key_1:
                    pos_str = 'l'
                elif col_0.nhbr_keys[(F+2)%4][u_idx] == col_key_1:
                    pos_str = 'u'
    else:
        # Dealing with cells
        cell_0 = col_0.cells[cell_key_0]
        cell_1 = col_1.cells[cell_key_1]
        
        [ndof_0] = cell_0.ndofs[:]
        lv_0 = cell_0.lv
        
        [ndof_1] = cell_1.ndofs[:]
        lv_1 = cell_1.lv
        
        if lv_1 - lv_0 == 0:
            pos_str = 's'
        else:
            [th0_0, th1_0] = cell_0.pos[:]
            [th0_1, th1_1] = cell_1.pos[:]
            
            mid_0 = (th0_0 + th1_0) / 2.
            mid_1 = (th1_0 + th1_1) / 2.

            if lv_1 - lv_0 == 1:
                # Cell 1 more refined
                if (mid_0 > mid_1):
                    pos_str = 'l'
                else:
                    pos_str = 'u'
            elif lv_1 - lv_1 == -1:
                # Cell 1 less refined
                if (mid_0 > mid_1):
                    pos_str = 'u'
                else:
                    pos_str = 'l'
            
    return proj.get_f2f_matrix(dim_str, ndof_0, ndof_1, lv_0, lv_1, pos_str)
