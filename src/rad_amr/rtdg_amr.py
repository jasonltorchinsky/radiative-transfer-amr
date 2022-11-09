import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, block_diag
from scipy.sparse.linalg import gmres
from scipy.linalg import eig
import sys
from time import perf_counter

from .Projection import Projection

sys.path.append('/quadrature')
import quadrature as qd
sys.path.append('/mesh')
from mesh import ji_mesh, tools

import matplotlib.pyplot as plt

'''
Matrices to check for initial solve:

'''

'''
Matrices verified for initial solve: *+1 means checked for non-uniform mesh
M_mass +1
M_scat +1
M_conv_loc +1
M_conv_bdry_1 +1
M_conv_bdry_2 + 1
'''


def rtdg_amr(mesh, uh_init, kappa, sigma, source, bdry_top, bdry_bot,
             max_iters, err_thresh, scat_type, **kwargs):
    '''
    Solve the radiative transfer problem.
    '''

    # Don't really know what this is, to be honest
    max_tab_deg = 20
    M_convs = []
    for cc in range(0, max_tab_deg):
        [nodes_x, _, _, _, _, _] = qd.quad_xya(cc + 1, 1, 1)
        M_conv_loc = np.transpose(np.asarray(qd.gl_ddx(nodes_x)))
        M_convs.append(M_conv_loc)

    uh = uh_init

    # Construct the mass matrix
    if kwargs['verbose']:
        print('Constructing the mass matrix...')
        t_start = perf_counter()
    
    M_mass = calc_mass_matrix(mesh, uh, kappa)

    sys.exit(2)
'''
    # Construct the scattering matrix
    if kwargs['verbose']:
        t_end = perf_counter()
        print('Constructed mass matrix in {:.4f} seconds!\n'.format(t_end - t_start))
        print('Constructing the scattering matrix...')
        t_start = perf_counter()
        
    M_scat = calc_scat_matrix(mesh, uh, sigma)

    # Construct conv_loc matrix
    if kwargs['verbose']:
        t_end = perf_counter()
        print('Constructed scattering matrix in {:.4f} seconds!\n'.format(t_end - t_start))
        print('Constructing the local convection matrix...')
        t_start = perf_counter()
        
    ilist_1 = np.zeros([ndof * nelts_a * dof_a], dtype = np.int32)
    jlist_1 = np.zeros([ndof * nelts_a * dof_a], dtype = np.int32)
    vlist_1 = np.zeros([ndof * nelts_a * dof_a])
    cnt_1   = 0

    ilist_2 = np.zeros([ndof * nelts_a * dof_a], dtype = np.int32)
    jlist_2 = np.zeros([ndof * nelts_a * dof_a], dtype = np.int32)
    vlist_2 = np.zeros([ndof * nelts_a * dof_a])
    cnt_2   = 0
    
    for key in sorted(mesh.ijlv.keys()): # Should contain only leafs
        dof_x = mesh.dof_x[key]
        dof_y = mesh.dof_y[key]
        [x0, y0, x1, y1] = mesh.pos[key]
        dx = x1 - x0
        dy = y1 - y0

        st_idx = uh.st_idxs[key]

        def idx(ii, jj, kk, ll):
            val = ll * dof_x * dof_y * nelts_a \
                + kk * dof_x * dof_y \
                + jj * dof_x \
                + ii
            return val

        [_, weights_x, _, weights_y, nodes_a, weights_a] = \
            qd.quad_xya(dof_x, dof_y, dof_a)

        for ii in range(0, dof_x):
            for jj in range(0, dof_y):
                for kk in range(0, nelts_a):
                    for ll in range(0, dof_a):
                        for ii_ii in range(0, dof_x):
                            ilist_1[cnt_1] = st_idx \
                                + idx(ii_ii, jj, kk, ll)
                            jlist_1[cnt_1] = st_idx \
                                + idx(ii, jj, kk, ll)
                            vlist_1[cnt_1] = (da / 2) \
                                * weights_x[ii] * weights_y[jj] \
                                * weights_a[ll] \
                                * (dy / 2) \
                                * M_convs[dof_x-1][ii_ii, ii] \
                                * np.cos(theta(kk, nodes_a[ll]))
                            cnt_1 += 1

                        for jj_jj in range(0, dof_y):
                            ilist_2[cnt_2] = st_idx \
                                + idx(ii, jj_jj, kk, ll)
                            jlist_2[cnt_2] = st_idx \
                                + idx(ii, jj, kk, ll)
                            vlist_2[cnt_2] = (da / 2) \
                                * weights_x[ii] * weights_y[jj] \
                                * weights_a[ll] \
                                * (dx / 2) \
                                * M_convs[dof_y-1][jj_jj, jj] \
                                * np.sin(theta(kk, nodes_a[ll]))
                            cnt_2 += 1
                        
                                
    M_conv_loc = coo_matrix((vlist_1, (ilist_1, jlist_1))).asformat('csr') \
        + coo_matrix((vlist_2, (ilist_2, jlist_2))).asformat('csr')

    # Boundary convection matrix, assume fixed nelts_a, dof_a
    if kwargs['verbose']:
        t_end = perf_counter()
        print('Constructed local convection matrix in {:.4f} seconds!\n'.format(t_end - t_start))
        print('Constructing the boundary convection matrices...')
        t_start = perf_counter()
        
    faces = ['r', 't', 'l', 'b']
    angle_fp = np.zeros([nelts_a, 4, dof_a])
    angle_fm = np.zeros([nelts_a, 4, dof_a])
    [_, _, _, _, nodes_a, weights_a] = qd.quad_xya(1, 1, dof_a)
    for kk in range(0, nelts_a):
        for face_idx in range(0, 4):
            face = faces[face_idx]
            if face == 'r':
                nm_x = 1
                nm_y = 0
            elif face == 't':
                nm_x = 0
                nm_y = 1
            elif face == 'l':
                nm_x = -1
                nm_y = 0
            elif face == 'b':
                nm_x = 0
                nm_y = -1
            for ll in range(0, dof_a):

                temp = np.cos(theta(kk, nodes_a[ll])) * nm_x \
                    + np.sin(theta(kk, nodes_a[ll])) * nm_y
                angle_fp[kk, face_idx, ll] = da / 2 * weights_a[ll] \
                    * temp * int(temp >= 0)
                angle_fm[kk, face_idx, ll] = da / 2 * weights_a[ll] \
                    * temp * int(temp <= 0)
                
    # First boundary convection matrix is for the inside of each element
    M_conv_bdry_1 = csr_matrix(np.zeros([ndof, ndof]))
    ilist = np.zeros([ndof], dtype = np.int32)
    jlist = np.zeros([ndof], dtype = np.int32)
    vlist = np.zeros([ndof])
    for face_idx in range(0, 4):
        face = faces[face_idx]
        cnt = 0
        for key in sorted(mesh.ijlv.keys()): # Should contain only leafs
            dof_x = mesh.dof_x[key]
            dof_y = mesh.dof_y[key]
            [x0, y0, x1, y1] = mesh.pos[key]
            dx = x1 - x0
            dy = y1 - y0

            st_idx = uh.st_idxs[key]

            def idx(ii, jj, kk, ll):
                val = ll * dof_x * dof_y * nelts_a \
                    + kk * dof_x * dof_y \
                    + jj * dof_x \
                    + (ii + 1)
                return val

            [_, weights_x, _, weights_y, _, _] = \
                qd.quad_xya(dof_x, dof_y, dof_a)

            p_list = [[dof_x - 1], range(0, dof_x), [0], range(0, dof_x)]
            q_list = [range(0, dof_y), [dof_y - 1], range(0, dof_y), [0]]

            for pp in p_list[face_idx]:
                for qq in q_list[face_idx]:
                    for kk in range(0, nelts_a):
                        for ll in range(0, dof_a):
                            ilist[cnt] = st_idx \
                                + idx(pp, qq, kk, ll) - 1
                            jlist[cnt] = ilist[cnt]
                            if (face == 'r' or face == 'l'):
                                temp = weights_y[qq] * dy / 2
                            elif (face == 't' or face == 'b'):
                                temp = weights_x[pp] * dx / 2
                            vlist[cnt] = temp * angle_fp[kk, face_idx, ll]
                            cnt += 1
                            
        M_conv_bdry_1 += csr_matrix((vlist, (ilist, jlist)),
                                    shape = [ndof, ndof])

    # Second boundary convection matrix communicates beween elements
    [_, tablist_f2f] = qd.face_2_face([], 1, 1)
    [_, _, tablist_f2df] = qd.face_2_dface([], 1, 1)
    M_conv_bdry_2 = csr_matrix(np.zeros([ndof, ndof]))
    for face_idx in range(0, 4):
        face = faces[face_idx]
        cnt = 0
        
        ilist = np.zeros([2 * ndof], dtype = np.int32)
        jlist = np.zeros([2 * ndof], dtype = np.int32)
        vlist = np.zeros([2 * ndof])
        
        for key, is_lf in sorted(mesh.is_lf.items()): # Should contain only leafs
            if is_lf:
                ijlv = mesh.ijlv[key]
                dof_x = mesh.dof_x[key]
                dof_y = mesh.dof_y[key]
                [x0, y0, x1, y1] = mesh.pos[key]
                dx = x1 - x0
                dy = y1 - y0
                
                st_idx = uh.st_idxs[key]
                
                def idx(ii, jj, kk, ll):
                    val = ll * dof_x * dof_y * nelts_a \
                        + kk * dof_x * dof_y \
                        + jj * dof_x \
                        + ii
                    return val
                
                [_, weights_x, _, weights_y, _, _] = \
                    qd.quad_xya(dof_x, dof_y, dof_a)
                
                [flag, nhbr_1, nhbr_2] = ji_mesh.fnd_nhbr(mesh, ijlv, face)
                
                p_list = [[dof_x - 1], range(0, dof_x), [0], range(0, dof_x)]
                q_list = [range(0, dof_y), [dof_y - 1], range(0, dof_y), [0]]
                
                if flag == 'f0':
                    key_nhbr = ji_mesh.get_key(nhbr_1)
                    dof_x_nhbr = mesh.dof_x[key_nhbr]
                    dof_y_nhbr = mesh.dof_y[key_nhbr]
                    
                    st_idx_nhbr = uh.st_idxs[key_nhbr]
                    
                    def idx_nhbr(ii, jj, kk, ll):
                        val = ll * dof_x_nhbr * dof_y_nhbr * nelts_a \
                            + kk * dof_x_nhbr * dof_y_nhbr \
                            + jj * dof_x_nhbr \
                            + ii
                        return val

                    p_list_nhbr = [[0], range(0, dof_x_nhbr),
                                   [dof_x_nhbr - 1], range(0, dof_x_nhbr)]
                    q_list_nhbr = [range(0, dof_y_nhbr), [0],
                                   range(0, dof_y_nhbr), [dof_y_nhbr - 1]]

                    if (face == 'r' or face == 'l'):
                        [tab_nhbr2me, tablist_f2f] = \
                            qd.face_2_face(tablist_f2f, dof_y_nhbr, dof_y)
                    elif (face == 't' or face == 'b'):
                        [tab_nhbr2me, tablist_f2f] = \
                            qd.face_2_face(tablist_f2f, dof_x_nhbr, dof_x)
                        
                    for kk in range(0, nelts_a):
                        for ll in range(0, dof_a):
                            for pp in p_list[face_idx]:
                                for qq in q_list[face_idx]:
                                    for pp_nhbr in p_list_nhbr[face_idx]:
                                        for qq_nhbr in q_list_nhbr[face_idx]:
                                            ilist[cnt] = st_idx \
                                                + idx(pp, qq, kk, ll)
                                            jlist[cnt] = st_idx_nhbr \
                                                + idx_nhbr(pp_nhbr, qq_nhbr, kk, ll)
                                            
                                            if (face == 'r' or face == 'l'):
                                                temp = weights_y[qq] * dy / 2 \
                                                    * tab_nhbr2me[qq, qq_nhbr]
                                            elif (face == 't' or face == 'b'):
                                                temp = weights_x[pp] * dx / 2 \
                                                    * tab_nhbr2me[pp, pp_nhbr]
                                            else:
                                                print('ERROR: Unidentifiable face. Aborting...')
                                                sys.exit(12)
                                            vlist[cnt] = temp \
                                                * angle_fm[kk, face_idx, ll]
                                            
                                            cnt += 1
                                            
                elif flag == 'cc':
                    key_nhbr_1 = ji_mesh.get_key(nhbr_1)
                    st_idx_nhbr_1 = uh.st_idxs[key_nhbr_1]
                    
                    key_nhbr_2 = ji_mesh.get_key(nhbr_2)
                    st_idx_nhbr_2 = uh.st_idxs[key_nhbr_2]
                    
                    dof_x_nhbr = mesh.dof_x[key_nhbr_1]
                    dof_y_nhbr = mesh.dof_y[key_nhbr_1]
                    
                    def idx_nhbr(ii, jj, kk, ll):
                        val = ll * dof_x_nhbr * dof_y_nhbr * nelts_a \
                            + kk * dof_x_nhbr * dof_y_nhbr \
                            + jj * dof_x_nhbr \
                            + ii
                        return val
                
                    p_list_nhbr = [[0], range(0, dof_x_nhbr),
                                   [dof_x_nhbr - 1], range(0, dof_x_nhbr)]
                    q_list_nhbr = [range(0, dof_y_nhbr), [0],
                                   range(0, dof_y_nhbr), [dof_y_nhbr - 1]]
                
                    if (face == 'r' or face == 'l'):
                        [_, tab_nhbr2me, tablist_f2df] = \
                            qd.face_2_dface(tablist_f2df, dof_y, dof_y_nhbr)
                    elif (face == 't' or face == 'b'):
                        [_, tab_nhbr2me, tablist_f2df] = \
                            qd.face_2_dface(tablist_f2df, dof_x, dof_x_nhbr)
                        
                    for kk in range(0, nelts_a):
                        for ll in range(0, dof_a):
                            for pp in p_list[face_idx]:
                                for qq in q_list[face_idx]:
                                    for pp_nhbr in p_list_nhbr[face_idx]:
                                        for qq_nhbr in q_list_nhbr[face_idx]:
                                            ilist[cnt] = st_idx \
                                                + idx(pp, qq, kk, ll) 
                                            jlist[cnt] = st_idx_nhbr_1 \
                                                + idx_nhbr(pp_nhbr, qq_nhbr, kk, ll)
                                            
                                            if (face == 'r' or face == 'l'):
                                                temp = weights_y[qq] * dy / 2 \
                                                    * tab_nhbr2me[qq, qq_nhbr]
                                            elif (face == 't' or face == 'b'):
                                                temp = weights_x[pp] * dx / 2 \
                                                    * tab_nhbr2me[pp, pp_nhbr]
                                            vlist[cnt] = temp \
                                                * angle_fm[kk, face_idx, ll]
                                            
                                            cnt += 1
                                            
                                            ilist[cnt] = st_idx \
                                                + idx(pp, qq, kk, ll)
                                            jlist[cnt] = st_idx_nhbr_2 \
                                                + idx_nhbr(pp_nhbr, qq_nhbr, kk, ll)
                                            
                                            end = np.shape(tab_nhbr2me)[1]
                                            if (face == 'r' or face == 'l'):
                                                temp = weights_y[qq] * dy / 2 \
                                                    * tab_nhbr2me[qq, qq_nhbr + int(end/2)]
                                            elif (face == 't' or face == 'b'):
                                                temp = weights_x[pp] * dx / 2 \
                                                    * tab_nhbr2me[pp, pp_nhbr + int(end/2)]
                                            else:
                                                print('ERROR: Unidentifiable face. Aborting...')
                                                sys.exit(12)
                                                
                                            vlist[cnt] = temp \
                                                * angle_fm[kk, face_idx, ll]
                                            
                                            cnt += 1
                                            
                elif (flag == 'pm' or flag == 'pp'):
                    key_nhbr = ji_mesh.get_key(nhbr_1)
                    dof_x_nhbr = mesh.dof_x[key_nhbr]
                    dof_y_nhbr = mesh.dof_y[key_nhbr]
                    
                    st_idx_nhbr = uh.st_idxs[key_nhbr]
                    
                    def idx_nhbr(ii, jj, kk, ll):
                        val = ll * dof_x_nhbr * dof_y_nhbr * nelts_a \
                            + kk * dof_x_nhbr * dof_y_nhbr \
                            + jj * dof_x_nhbr \
                            + ii
                        return val
                    
                    p_list_nhbr = [[0], range(0, dof_x_nhbr),
                                   [dof_x_nhbr - 1], range(0, dof_x_nhbr)]
                    q_list_nhbr = [range(0, dof_y_nhbr), [0],
                                   range(0, dof_y_nhbr), [dof_y_nhbr - 1]]
                    
                    if (face == 'r' or face == 'l'):
                        [tab_nhbr2me, _, tablist_f2df] = \
                            qd.face_2_dface(tablist_f2df, dof_y_nhbr, dof_y)
                    elif (face == 't' or face == 'b'):
                        [tab_nhbr2me, _, tablist_f2df] = \
                            qd.face_2_dface(tablist_f2df, dof_x_nhbr, dof_x)
                        
                    for kk in range(0, nelts_a):
                        for ll in range(0, dof_a):
                            for pp in p_list[face_idx]:
                                for qq in q_list[face_idx]:
                                    for pp_nhbr in p_list_nhbr[face_idx]:
                                        for qq_nhbr in q_list_nhbr[face_idx]:
                                            ilist[cnt] = st_idx \
                                                + idx(pp, qq, kk, ll)
                                            jlist[cnt] = st_idx_nhbr \
                                                + idx_nhbr(pp_nhbr, qq_nhbr, kk, ll)
                                            
                                            end = np.shape(tab_nhbr2me)[0]
                                            if (face == 'r' or face == 'l'):
                                                if flag == 'pm':
                                                    temp = weights_y[qq] * dy / 2 \
                                                        * tab_nhbr2me[qq, qq_nhbr]
                                                elif flag == 'pp':
                                                    temp = weights_y[qq] * dy / 2 \
                                                        * tab_nhbr2me[qq + int(end/2), qq_nhbr]
                                            elif (face == 't' or face == 'b'):
                                                if flag == 'pm':
                                                    temp = weights_x[pp] * dx / 2 \
                                                        * tab_nhbr2me[pp, pp_nhbr]
                                                elif flag == 'pp':
                                                    temp = weights_x[pp] * dx / 2 \
                                                        * tab_nhbr2me[pp + int(end/2), pp_nhbr]
                                            else:
                                                print('ERROR: Unidentifiable face. Aborting...')
                                                sys.exit(12)
                                                
                                            vlist[cnt] = temp \
                                                * angle_fm[kk, face_idx, ll]
                                            
                                            cnt += 1
                

        M_conv_bdry_2 += csr_matrix((vlist, (ilist, jlist)),
                                    shape = [ndof, ndof])

    M_conv = M_conv_bdry_1 + M_conv_bdry_2 - M_conv_loc

    # Create forcing term and boundary DOFs
    if kwargs['verbose']:
        t_end = perf_counter()
        print('Constructed the boundary convection matrices in {:.4f} seconds!\n'.format(t_end - t_start))
        print('Constructing the forcing terms matrix...')
        t_start = perf_counter()
        
    sch = np.zeros([ndof])
    for key, is_lf in sorted(mesh.is_lf.items()): # Should contain only leafs
        if is_lf:
            dof_x = mesh.dof_x[key]
            dof_y = mesh.dof_y[key]
            [x0, y0, x1, y1] = mesh.pos[key]
            dx = x1 - x0
            dy = y1 - y0
            
            st_idx = uh.st_idxs[key]
            
            [nodes_x, weights_x, nodes_y, weights_y, nodes_a, weights_a] = \
                qd.quad_xya(dof_x, dof_y, dof_a)
            
            temp = np.zeros([dof_x, dof_y, nelts_a, dof_a])
            for ii in range(0, dof_x):
                x = x0 + (x1 - x0) / 2 * (nodes_x[ii] + 1)
                for jj in range(0, dof_y):
                    y = y0 + (y1 - y0) / 2 * (nodes_y[jj] + 1)
                    for kk in range(0, nelts_a):
                        for ll in range(0, dof_a):
                            a = kk * da + (nodes_a[ll] + 1) / 2 * da
                            
                            temp[ii, jj, kk, ll] = (dx / 2) * (dy / 2) \
                                * (da / 2) * source(x, y, a) \
                                * weights_x[ii] * weights_y[jj] * weights_a[ll]
            sch[st_idx: st_idx + dof_x * dof_y * nelts_a * dof_a] = \
                temp.flatten(order = 'F')

    dof_top = []
    dof_bot = []
    for key, is_lf in sorted(mesh.is_lf.items()): # Should contain only leafs
        if is_lf:
            ijlv = mesh.ijlv[key]
            dof_x = mesh.dof_x[key]
            dof_y = mesh.dof_y[key]
            [x0, y0, x1, y1] = mesh.pos[key]
            dx = x1 - x0
            dy = y1 - y0
            
            st_idx = uh.st_idxs[key]
            
            def idx(ii, jj, kk, ll):
                val = ll * dof_x * dof_y * nelts_a \
                    + kk * dof_x * dof_y \
                    + jj * dof_x \
                    + ii
                return val

            [nodes_x, _, nodes_y, _, nodes_a, _] = \
                qd.quad_xya(dof_x, dof_y, dof_a)
            
            def Fdx(x):
                return (x + 1) / 2 * dx + x0
            
            def Fdy(y):
                return (y + 1) / 2 * dy + y0
            
            [flag, _, _] = ji_mesh.fnd_nhbr(mesh, ijlv, 't')

            if flag == 'nn':
                for ii in range(0, dof_x):
                    for kk in range(int(nelts_a/2), nelts_a):
                        for ll in range(0, dof_a):
                            this_dof = st_idx + idx(ii, dof_y - 1, kk, ll)
                            dof_top.append(this_dof)
                            uh.coeffs[this_dof] = bdry_top( Fdx(nodes_x[ii]),
                                                            Fdy(nodes_y[dof_y - 1]),
                                                            theta(kk, nodes_a[ll]) )

            [flag, _, _] = ji_mesh.fnd_nhbr(mesh, ijlv, 'b')

            if flag == 'nn':
                for ii in range(0, dof_x):
                    for kk in range(0, int(nelts_a/2)):
                        for ll in range(0, dof_a):
                            this_dof = st_idx + idx(ii, 0, kk, ll)
                            dof_bot.append(this_dof)
                            uh.coeffs[this_dof] = bdry_bot( Fdx(nodes_x[ii]),
                                                            Fdy(nodes_y[0]),
                                                            theta(kk, nodes_a[ll]) )

    bdry = dof_top + dof_bot
    intr = list(set(range(0, ndof)).symmetric_difference(bdry))

    # Construct the total mass matrix and solve the system.
    if kwargs['verbose']:
        t_end = perf_counter()
        print('Constructed the forcing terms matrix in {:.4f} seconds!\n'.format(t_end - t_start))
        print('Constructing the total mass matrix and solving the system...')
        t_start = perf_counter()
        
    M_tot = M_mass + M_conv - M_scat
    ix_grid_ii = np.ix_(intr, intr)
    A = M_tot[ix_grid_ii]
    ix_grid_i  = np.ix_(intr)
    ix_grid_b  = np.ix_(bdry)
    ix_grid_ib = np.ix_(intr, bdry)
    # Something in my b is wrong
    b = sch[ix_grid_i] - M_tot[ix_grid_ib] @ np.asarray(uh.coeffs)[ix_grid_b]

    # Solve the linear system
    x, exit_code = gmres(A, b, tol = err_thresh, maxiter = max_iters)

    if exit_code > 0:
        print(( 'WARNING: Convergence to tolerance not achieved' +
                ' within desired number of iterations of GMRES. [rtdg_amr.py]\n' ))
    elif exit_code < 0:
        print(( 'WARNING: Illegal input of breakdown of GMRES. [rtdg_amr.py]\n' ))

    for ii in range(0, np.shape(x)[0]):
        uh.coeffs[ix_grid_i[0][ii]] = x[ii]
    
    if kwargs['verbose']:
        t_end = perf_counter()
        print('Solved the system in {:.4f} seconds!'.format(t_end - t_start))
        print('Ax = b exit code: {}.\n'.format(exit_code))
    
    return uh
'''
def f_scat(x, y, scat_type):

    # Scattering function, always do Rayleigh for now

    return 1. / (3. * np.pi) * (1 + np.cos(x - y)**2)

def calc_mass_matrix(mesh, uh, kappa):
    

def calc_mass_matrix_orig(mesh, uh, kappa):

    ndof = uh.ndof
    kappah = Projection(mesh = mesh, has_a = False, u = kappa)

    # Store local-column matrices in here
    ncols = len(list(mesh.cols.keys()))
    col_mtxs = ncols * [0]
    col_idx = 0
    
    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos
            dx = x1 - x0
            dy = y1 - y0
            
            st_kappah_idx = kappah.st_idxs[col_key]
            
            # Store local-element matrices in here
            ncells = len(list(col.cells.keys()))
            cell_mtxs = ncells * [0]
            cell_idx = 0
            
            for cell_key, cell in sorted(col.cells.items()):
                if cell.is_lf:
                    [dof_x, dof_y, dof_a] = cell.ndofs
                    
                    # We construct local-element matrices to put into a
                    # local-column matrix to put into a global matrix
                    cell_ndof = dof_x * dof_y * dof_a
                    ilist = np.zeros([cell_ndof], dtype = np.int32)
                    jlist = np.zeros([cell_ndof], dtype = np.int32)
                    vlist = np.zeros([cell_ndof])
                    cnt = 0
                    
                    [a0, a1] = cell.pos
                    da = a1 - a0
                    
                    def loc_elt_idx(ii, jj, aa):
                        val = dof_x * dof_y * aa \
                            + dof_x * jj \
                            + ii
                        return val
                    
                    st_uh_idx = uh.st_idxs[str([col_key, cell_key])]
                    
                    temp_kappah = kappah.coeffs[st_kappah_idx:st_kappah_idx + dof_x * dof_y]
                    temp_kappah = np.asarray(temp_kappah).reshape(dof_x, dof_y, order = 'F')
                    
                    [_, weights_x, _, weights_y, nodes_a, weights_a] = \
                        qd.quad_xya(dof_x, dof_y, dof_a)
                    
                    dcoeff = dx * dy * da / 8
                    for ii in range(0, dof_x):
                        wx_i = weights_x[ii]
                        for jj in range(0, dof_y):
                            wy_j = weights_y[jj]
                            kappa_ij = temp_kappah[ii, jj]
                            for aa in range(0, dof_a):
                                wth_a = weights_a[aa]
                                
                                ilist[cnt] = loc_elt_idx(ii, jj, aa)
                                jlist[cnt] = ilist[cnt]
                                
                                vlist[cnt] = dcoeff * wx_i * wy_j * wth_a * kappa_ij
                                cnt += 1
                                
                    cell_mtxs[cell_idx] = coo_matrix((vlist, (ilist, jlist)))
                    cell_idx += 1
                    
            col_mtxs[col_idx] = block_diag(cell_mtxs)
            col_idx += 1
                
    M_mass = block_diag(col_mtxs).asformat('csr')

    return M_mass

def calc_scat_matrix(mesh, uh, sigma):

    '''ndof = uh.ndof
    sigmah = Projection(mesh = mesh, has_a = False, u = sigma)

    ilist = np.zeros([ndof], dtype = np.int32)
    jlist = np.zeros([ndof], dtype = np.int32)
    vlist = np.zeros([ndof])
    cnt = 0
    for col_key, col in sorted(mesh.cols.items()):
        if col.is_lf:
            [x0, y0, x1, y1] = col.pos
            dx = x1 - x0
            dy = y1 - y0
            for cell_key, cell in sorted(col.cells.items()):
                if cell.is_lf:
                    [dof_x, dof_y, dof_a] = cell.ndofs
                    [a0, a1] = cell.pos
                    da = a1 - a0
                    
                    def local_elt_idx(ii, jj, aa):
                        val = dof_x * dof_y * aa\
                            + dof_x * jj \
                            + ii
                        return val

                    def pqr_to_ija(pp, qq, rr):
                        pp_pp = pp
                        

                    st_uh_idx = uh.st_idxs[str([col_key, cell_key])]
                    st_sigmah_idx = sigmah.st_idxs[col_key]
                    
                    temp_sigmah = sigmah.coeffs[st_sigmah_idx:st_sigmah_idx + dof_x * dof_y]
                    temp_sigmah = np.asarray(temp_sigmah).reshape(dof_x, dof_y, order = 'F')
        
                    [_, weights_x, _, weights_y, nodes_a, weights_a] = \
                        qd.quad_xya(dof_x, dof_y, dof_a)

                    for ii in range(0, dof_x):
                        for jj in range(0, dof_y):
                            for aa in range(0, dof_a):
                                for rr in range(0, dof_a):
                                    ilist[cnt] = st_uh_idx \
                                        + local_elt_idx(ii, jj, aa)
                                    jlist[cnt] = st_uh_idx \
                                        + idx(ii, jj, rr)
                                    vlist[cnt] = (da**2 / 4) \
                                        * weights_a[ll] * weights_a[ll_ll] \
                                        * f_scat( theta(kk_kk, nodes_a[ll_ll]),
                                                  theta(kk, nodes_a[ll]),
                                                  scat_type ) \
                                        * (dx * dy / 4) \
                                        * weights_x[ii] * weights_y[jj] \
                                        * temp_sigmah[ii, jj]
                                cnt += 1
                                
    M_scat = coo_matrix((vlist, (ilist, jlist))).asformat('csr')'''
