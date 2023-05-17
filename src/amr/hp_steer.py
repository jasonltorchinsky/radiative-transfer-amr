import numpy as np
from scipy.special import legendre

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
        wx  = wx.reshape([ndof_x, 1])
        wy  = wy.reshape([1, ndof_y])
        
        # uh_hat is the numerical solution integrated in angle
        uh_hat = intg_col_th(mesh, uh, col_key)

        # Calculate a_p^K,x, a_nx^K,x
        Ln = np.zeros([ndof_x, 1])
        for ii in range(0, ndof_x):
            Ln[ii, 0] = legendre(ndof_x)(xxb[ii])
            
        dofs_y = np.arange(0, ndof_y)
        Lq = np.zeros([ndof_y, ndof_y])
        for qq in range(0, ndof_y):
            for jj in range(0, ndof_y):
                Lq[qq, jj] = legendre(dofs_y[qq])(yyb[jj])
                
        anq = np.zeros([ndof_y])
        anx_sq = 0.
        for qq in range(0, ndof_y):
            anq[qq] = (2. * ndof_x + 1.) * (2. * qq + 1.) / 4. \
                * np.sum(wx * wy * uh_hat * Ln * Lq[qq, :])
            anx_sq += (anq[qq])**2 * (2. / (2. * qq + 1))

        # Calculate a_q^K,y, a_ny^K,y
        dofs_x = np.arange(0, ndof_x)
        Ln = np.zeros([ndof_x, ndof_x])
        for pp in range(0, ndof_x):
            for ii in range(0, ndof_x):
                Ln[pp, ii] = legendre(dofs_x[pp])(xxb[ii])
        
        Lq = np.zeros([1, ndof_y])
        for jj in range(0, ndof_y):
            Lq[0, jj] = legendre(ndof_y)(yyb[jj])
            
        apn = np.zeros([ndof_x])
        any_sq = 0.
        for pp in range(0, ndof_x):
            apn[qq] = (2. * pp + 1.) * (2. * ndof_y + 1.) / 4. \
                * np.sum(wx * wy * uh_hat * Ln[pp, :] * Lq)
            any_sq += (apn[pp])**2 * (2. / (2. * pp + 1))

        term_0 = np.log((2. * ndof_x + 1.) / (2 * anx_sq)) / (2. * np.log(ndof_x))
        term_1 = np.log((2. * ndof_y + 1.) / (2 * any_sq)) / (2. * np.log(ndof_y))
        lp = 0.5 * (term_0 + term_1)
        
        if lp - 0.5 >= 0.5 * (ndof_x * ndof_y) + 1.:
            ref_form = 'h'
        else:
            ref_form = 'p'
        
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
            
            
            # uh_hat is the numerical solution integrated in space
            uh_hat = intg_cell_xy(mesh, uh, col_key, cell_key)
            
            # Calculate a_p^K,th, a_nx^K,x
            Lnth = np.zeros([ndof_th])
            for aa in range(0, ndof_th):
                Lnth[aa] = legendre(ndof_th)(thb[aa])
            
            anth_sq = (((2. * ndof_th + 1.) / 2.) * np.sum(uh_hat * wth * Lnth))**2
            
            lp = np.log((2. * ndof_th + 1.) / (2 * anth_sq)) / (2. * np.log(ndof_th))
            
            if lp - 0.5 >= ndof_th + 1.:
                ref_form = 'h'
            else:
                ref_form = 'p'
            
            return ref_form
