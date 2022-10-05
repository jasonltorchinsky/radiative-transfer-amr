def pos2key(mesh, point):

    for key, is_lf in mesh.is_lf.items():
        if is_lf:
            pos = mesh.pos[key]
            
            within = mesh.ndim * [False]
            for dim in range(0, mesh.ndim):
                if (point[dim] > pos[dim]) \
                   and (point[dim] < pos[dim + mesh.ndim]):
                    within[dim] = True
                    
            if all(within):
                return key
            
    return None
