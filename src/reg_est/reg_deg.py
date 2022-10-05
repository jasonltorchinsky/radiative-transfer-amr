import numpy as np
from scipy.special import eval_legendre
import sys

sys.path.append('/quadrature')
import quadrature as qd
sys.path.append('/rad_amr')
import rad_amr as ar

def reg_deg(mesh, uh):

    # We want to take the mean with respest to angle. Assume not normalized
    if uh.has_a:
        uh_xy = uh.intg_a(mesh)
    else:
        uh_xy = uh
    uh_xy.coeffs = (np.array(uh_xy.coeffs) / (2 * np.pi)).tolist()

    max_dof = np.amax(np.array(list(mesh.dof_x.values())
                               + list(mesh.dof_y.values())))

    leg_vals = dict()

    for ii in range(0, max_dof):
        [nodes_x, _, _, _, _, _] = qd.quad_xya(ii + 1, 1, 1)
        nn = np.arange(0, max_dof)[:, np.newaxis]
        nn = np.tile(nn, (1, ii + 1)) 
        xx = np.tile(nodes_x, (max_dof, 1))
        leg_vals[ii] = eval_legendre(nn, xx)

    regth = ar.Projection(mesh = mesh, has_a = False)
    regmm = ar.Projection(mesh = mesh, has_a = False)
    for key, is_lf in sorted(mesh.is_lf.items()):
        if is_lf:
            dof_x = mesh.dof_x[key]
            dof_y = mesh.dof_y[key]
            
            st_idx = uh_xy.st_idxs[key]
            uh_xy_elt = uh_xy.coeffs[st_idx:st_idx + dof_x * dof_y]
            uh_xy_elt = np.array(uh_xy_elt).reshape(dof_x, dof_y, order = 'F')
            [theta_x, mm_x, theta_y, mm_y] = reg_idx(uh_xy_elt, leg_vals)

            theta = (theta_x + theta_y) / 2.
            mm = (mm_x + mm_y) / 2.

            # Something in here isn't quite right... Differs from Shukai's code.
            # I now believe this is a numerical instability
            regth.coeffs[st_idx:st_idx + dof_x * dof_y] = \
                (theta * np.ones([dof_x * dof_y])).tolist()
            regmm.coeffs[st_idx:st_idx + dof_x * dof_y] = \
                (mm * np.ones([dof_x * dof_y])).tolist()
            
    return [regth, regmm]
        

def reg_idx(uh_elt, leg_vals):

    [dof_x, dof_y] = np.shape(uh_elt)
    [_, weights_x, _, weights_y, _, _] = qd.quad_xya(dof_x, dof_y, 1)
    nx_leg = dof_x
    ny_leg = dof_y

    aa = np.zeros([nx_leg, ny_leg])
    for ii in range(0, nx_leg):
        for jj in range(0, ny_leg):
            temp = (leg_vals[nx_leg - 1][ii, :])[:, np.newaxis] \
                @ (leg_vals[ny_leg - 1][jj, :])[np.newaxis, :]
            temp = uh_elt * temp
            temp = weights_x[np.newaxis, :] @ temp @ weights_y[:, np.newaxis]
            aa[ii, jj] = temp[0,0] * (2 * ii + 1) * (2 * jj + 1) / 4

    mult = 2. / (1 + 2 * np.arange(0, nx_leg))
    aa_x = np.sqrt(aa**2 @ mult[:, np.newaxis])

    aa_tr = aa.transpose()
    mult = 2. / (1 + 2 * np.arange(0, ny_leg))
    aa_y = np.sqrt(aa_tr**2 @ mult[:, np.newaxis])

    [theta_x, mm_x] = reg_leg(nx_leg, aa_x)
    [theta_y, mm_y] = reg_leg(ny_leg, aa_y)

    return [theta_x, mm_x, theta_y, mm_y]

def reg_leg(nn, aa):

    if nn > np.shape(aa)[0]:
        # Some sort of error catch from original?
        print('ERROR: nn exceeds length of aa. Using nn = len(aa).')
        nn = np.shape(aa)[0]

    temp = np.abs(np.log(np.abs(aa))).flatten(order = 'F')
    coeff_1 = np.arange(0, nn) @ temp
    coeff_2 = np.sum(temp)

    temp = 6 * (2 * coeff_1 - (nn - 1) * coeff_2) / (nn * (nn**2 - 1))

    theta = np.exp(-temp)
    mm = np.log((2 * nn - 1) / (2 * aa[nn - 1]**2)) / (2 * np.log(nn - 1))

    return [theta, mm]
