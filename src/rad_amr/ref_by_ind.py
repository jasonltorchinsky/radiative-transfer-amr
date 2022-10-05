import numpy as np

def ref_by_ind(mesh, ind, ref_ratio, regth, regmm):
    '''
    Refine the mesh by some index (ind).
    '''

    sorted_ind = []
    for key in sorted(ind.st_idxs.keys()):
        st_idx = ind.st_idxs[key]
        sorted_ind.append(list([key, float(ind.coeffs[st_idx])]))
    
    sorted_ind.sort(key = lambda x: x[1])
    ref_st_idx = int(np.floor((1 - ref_ratio) * len(ind.st_idxs.keys()))) - 1
    to_ref_keys = np.asarray(sorted_ind, dtype = np.int32)[ref_st_idx:, 0]
    for key in sorted(list(to_ref_keys)):
        ijlv = mesh.ijlv[key]
        dof_xy = (mesh.dof_x[key] + mesh.dof_y[key]) / 2.
        mm_idx = regmm.st_idxs[key]
        mm = regmm.coeffs[mm_idx]
        if dof_xy <= (mm - 1/2):
            mesh.ref_cell(ijlv, kind = 'p')
        else:
            mesh.ref_cell(ijlv, kind = 'h')
    

    return mesh


    
