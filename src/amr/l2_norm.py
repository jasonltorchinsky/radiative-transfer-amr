import numpy as np
import sys

sys.path.append('/quadrature')
import quadrature as qd

def l2_norm_xya(mesh, u_proj):

    norm = 0
    for key, pos in sorted(mesh.pos.items()): # Should only conain leafs!
        [x0, y0, x1, y1] = pos
        dof_x   = mesh.dof_x[key]
        dof_y   = mesh.dof_y[key]
        dof_a   = mesh.dof_a[key]
        nelts_a = mesh.nelts_a[key]

        dx = x1 - x0
        dy = y1 - y0
        da = 2 * np.pi / nelts_a

        [_, weights_x, _, weights_y, _, weights_a] \
            = qd.quad_xya(dof_x, dof_y, dof_a)

        u_proj_elt = u_proj.proj[key]
        u_proj_elt = u_proj_elt**2
        u_proj_elt.reshape([dof_x * dof_y * nelts_a, dof_a])
        u_proj_elt = u_proj_elt @ np.asarray(weights_a) * da / 2
        u_proj_elt.reshape([dof_x * dof_y, nelts_a])
        u_proj_elt = np.sum(u_proj_elt, axis = 1)
        u_proj_elt.reshape([dof_x, dof_y])
        u_proj_elt = u_proj_elt @ np.asarray(weights_y) * dy / 2
        u_proj_elt = u_proj_elt @ np.asarray(weights_x) * dx / 2
        norm += u_proj_elt

    return np.sqrt(norm)
