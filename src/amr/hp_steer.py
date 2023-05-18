import numpy as np
from scipy.special import eval_legendre

import dg.quadrature as qd
from dg.projection import intg_cell_xy, intg_col_th

def hp_steer_col(mesh, uh, col_key):
    col = mesh.cols[col_key]
    
    if col.is_lf:
        [x0, y0, x1, y1]   = col.pos[:]
        [dx, dy]           = [x1 - x0, y1 - y0]
        [ndof_x, ndof_y]   = col.ndofs[:]

        [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = ndof_x,
                                                nnodes_y = ndof_y)
        xxb = xxb.reshape([1, ndof_x])
        wx  =  wx.reshape([ndof_x, 1])
        yyb = yyb.reshape([1, ndof_y])
        wy  =  wy.reshape([1, ndof_y])
        
        # uh_hat is the numerical solution integrated in angle
        uh_hat = intg_col_th(mesh, uh, col_key)
        
        wx_wy_uh_hat = wx * wy * uh_hat
        
        # Calculate a_p^K,x, a_nx^K,x
        L_nxm = eval_legendre(ndof_x, xxb)
        ndofs_y = np.arange(0, ndof_y).reshape([1, ndof_y])
        L_jn = eval_legendre(ndofs_y.transpose(), yyb)
        
        a_nxj = (2. * ndof_x + 1.) * (2. * ndofs_y + 1.) / 4. \
            * L_nxm @ wx_wy_uh_hat @ L_jn

        ax_nx_sq = 0
        for jj in range(0, ndof_y):
            ax_nx_sq += (a_nxj[0, jj])**2 * (2. / (2. * jj + 1.))

        # Calculate a_q^K,y, a_ny^K,y
        ndofs_x = np.arange(0, ndof_x).reshape([ndof_x, 1])
        L_im = eval_legendre(ndofs_x, xxb)
        L_nyn = eval_legendre(ndof_y, yyb)
        
        a_iny = (2. * ndofs_x + 1.) * (2. * ndof_y + 1.) / 4. \
            * L_im @ wx_wy_uh_hat @ L_nyn.transpose()

        ay_ny_sq = 0
        for ii in range(0, ndof_x):
            ay_ny_sq += (a_iny[ii, 0])**2 * (2. / (2. * ii + 1.))

        term_0 = np.log((2. * ndof_x + 1.) / (2 * ax_nx_sq)) / (2. * np.log(ndof_x))
        term_1 = np.log((2. * ndof_y + 1.) / (2 * ay_ny_sq)) / (2. * np.log(ndof_y))
        lp = 0.5 * (term_0 + term_1)
        
        if lp - 0.5 >= 0.5 * (ndof_x + ndof_y) + 1.:
            ref_form = 'h'
        else:
            ref_form = 'p'
        
        print('spt ref form: {}, {:05.2f} >?< {}'.format(ref_form,
                                                         lp - 0.5,
                                                         0.5 * (ndof_x + ndof_y) + 1.))
            
        return ref_form

def hp_steer_cell(mesh, uh, col_key, cell_key):
    col = mesh.cols[col_key]
    
    if col.is_lf:
        cell = col.cells[cell_key]
        if cell.is_lf:
            [th0, th1] = cell.pos[:]
            dth        = th1 - th0
            [ndof_th]  = cell.ndofs[:]
            
            [_, _, _, _, thb, wth] = qd.quad_xyth(nnodes_th = ndof_th)
            
            thb = thb.reshape([1, ndof_th])
            wth = wth.reshape([1, ndof_th])
            
            # uh_hat is the numerical solution integrated in space
            uh_hat = intg_cell_xy(mesh, uh, col_key, cell_key).reshape([1, ndof_th])
            
            # Calculate a_p^K,th, a_nx^K,x
            L_nthr = eval_legendre(ndof_th, thb)
            a_nth = (2. * ndof_th + 1.) / 2. \
                * L_nthr.transpose() @ (wth * uh_hat)
            ath_nth_sq = (a_nth[0, 0])**2 * (2. / (2. * ndof_th + 1.))
            
            lp = np.log((2. * ndof_th + 1.) / (2 * ath_nth_sq)) \
                / (2. * np.log(ndof_th))
            
            if lp - 0.5 >= ndof_th + 1.:
                ref_form = 'h'
            else:
                ref_form = 'p'

            print('ang ref form: {}, {:05.2f} >?< {}'.format(ref_form,
                                                             lp - 0.5,
                                                             ndof_th + 1))
            
            return ref_form
