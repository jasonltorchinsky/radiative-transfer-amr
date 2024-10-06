import numpy as np

import dg.mesh       as ji_mesh
import dg.projection as proj
import dg.quadrature as qd

from .Error_Indicator import Error_Indicator
from .hp_steer        import hp_steer_col

def col_jump_err(mesh, uh_proj, **kwargs):

    default_kwargs = {"ref_col"      : True,
                      "col_ref_form" : "hp",
                      "col_ref_kind" : "spt",
                      "col_ref_tol"  : 0.85,
                      "ref_cell"      : False,
                      "cell_ref_form" : None,
                      "cell_ref_kind" : None,
                      "cell_ref_tol"  : None}
    kwargs = {**default_kwargs, **kwargs}
    
    err_ind = Error_Indicator(mesh, **kwargs)
    col_nref = 0
    
    col_items = sorted(mesh.cols.items())

    rng = np.random.default_rng()
    
    # Track maximum error(s) to calculate hp-steering only where needed
    col_max_err  = 0.
    col_ref_tol  = kwargs["col_ref_tol"]
    
    # We get the jumps for each pair of neighboring cells
    # _0 refers to self, _1 refers to neighbor
    # _A => smaller spatial domain, _B => larger number of spatial DoFs
    # _G => smaller angular domain, _H => larger number of angular DoFs
    for col_key_0, col_0 in col_items:
        if col_0.is_lf:
            # Get self-column info
            [x0_0, y0_0, x1_0, y1_0] = col_0.pos[:]
            [dx_0, dy_0]             = [x1_0 - x0_0, y1_0 - y0_0]
            perim                    = 2. * dx_0 + 2. * dy_0
            [nx_0, ny_0]             = col_0.ndofs[:]
            
            [xxb_0, _, yyb_0, _, _, _] = qd.quad_xyth(nnodes_x = nx_0,
                                                      nnodes_y = ny_0)
            
            cell_items_0 = sorted(col_0.cells.items())
            
            # Loop through faces to calculate error
            col_jump = 0.
            for F in range(0, 4):
                nhbr_keys = list(set(col_0.nhbr_keys[F]))
                
                for col_key_1 in nhbr_keys:
                    if col_key_1 is not None:
                        col_1 = mesh.cols[col_key_1]
                        if col_1.is_lf:
                            # Get nhbr-column info
                            [x0_1, y0_1, x1_1, y1_1] = col_1.pos[:]
                            [dx_1, dy_1]             = [x1_1 - x0_1, y1_1 - y0_1]
                            [nx_1, ny_1]             = col_1.ndofs[:]
                            
                            [xxb_1, _, yyb_1, _, _, _] = qd.quad_xyth(nnodes_x = nx_1,
                                                                      nnodes_y = ny_1)
                            
                            # Get integration interval, quadrature info
                            if (F%2 == 0):
                                if dy_0 <= dy_1:
                                    [y0_A, y1_A] = [y0_0, y1_0]
                                    [dy_A]       = [dy_0]
                                else:
                                    [y0_A, y1_A] = [y0_1, y1_1]
                                    [dy_A]       = [dy_1]
                                    
                                if ny_0 >= ny_1:
                                    [ny_B]       = [ny_0]
                                else:
                                    [ny_B]       = [ny_1]
                                    
                                [_, _, yyb_B, wy_B, _, _] = qd.quad_xyth(nnodes_y = ny_B)
                                wy_B = wy_B.reshape([ny_B, 1])
                                yyf_A   = proj.push_forward(y0_A, y1_A, yyb_B)
                                yyb_A_0 = proj.pull_back(y0_0, y1_0, yyf_A)
                                yyb_A_1 = proj.pull_back(y0_1, y1_1, yyf_A)
                                
                                psi_0_mtx = np.zeros([ny_0, ny_B])
                                for jj in range(0, ny_0):
                                    for jj_p in range(0, ny_B):
                                        psi_0_mtx[jj, jj_p] = qd.lag_eval(yyb_0, jj, yyb_A_0[jj_p])
                                        
                                psi_1_mtx = np.zeros([ny_1, ny_B])
                                for qq in range(0, ny_1):
                                    for jj_p in range(0, ny_B):
                                        psi_1_mtx[qq, jj_p] = qd.lag_eval(yyb_1, qq, yyb_A_1[jj_p])
                            else: # (F%2 == 1)
                                if dx_0 <= dx_1:
                                    [x0_A, x1_A] = [x0_0, x1_0]
                                    [dx_A]       = [dx_0]
                                else:
                                    [x0_A, x1_A] = [x0_1, x1_1]
                                    [dx_A]       = [dx_1]
                                    
                                if nx_0 >= nx_1:
                                    [nx_B]       = [nx_0]
                                else:
                                    [nx_B]       = [nx_1]
                                    
                                [xxb_B, wx_B, _, _, _, _] = qd.quad_xyth(nnodes_x = nx_B)
                                wx_B    = wx_B.reshape([nx_B, 1])
                                xxf_A   = proj.push_forward(x0_A, x1_A, xxb_B)
                                xxb_A_0 = proj.pull_back(x0_0, x1_0, xxf_A)
                                xxb_A_1 = proj.pull_back(x0_1, x1_1, xxf_A)
                                
                                phi_0_mtx = np.zeros([nx_0, nx_B])
                                for ii in range(0, nx_0):
                                    for ii_p in range(0, nx_B):
                                        phi_0_mtx[ii, ii_p] = qd.lag_eval(xxb_0, ii, xxb_A_0[ii_p])
                                        
                                phi_1_mtx = np.zeros([nx_1, nx_B])
                                for pp in range(0, nx_1):
                                    for ii_p in range(0, nx_B):
                                        phi_1_mtx[pp, ii_p] = qd.lag_eval(xxb_1, pp, xxb_A_1[ii_p])
                                        
                            # Loop through self-cells
                            for cell_key_0, cell_0 in cell_items_0:
                                if cell_0.is_lf:
                                    # Get self-cell info
                                    [th0_0, th1_0] = cell_0.pos[:]
                                    [dth_0]        = [th1_0 - th0_0]
                                    [nth_0]        = cell_0.ndofs[:]
                                    
                                    [_, _, _, _, thb_0, _] = qd.quad_xyth(nnodes_th = nth_0)
                                    
                                    # Get solution values in cell 0
                                    if F == 0:
                                        uh_0 = uh_proj.cols[col_key_0].cells[cell_key_0].vals[-1,:,:]
                                    elif F == 1:
                                        uh_0 = uh_proj.cols[col_key_0].cells[cell_key_0].vals[:,-1,:]
                                    elif F == 2:
                                        uh_0 = uh_proj.cols[col_key_0].cells[cell_key_0].vals[0,:,:]
                                    else: # F == 3
                                        uh_0 = uh_proj.cols[col_key_0].cells[cell_key_0].vals[:,0,:]
                                    
                                    # Get nhbr_cell info
                                    nhbr_keys = \
                                        ji_mesh.get_cell_nhbr_in_col(mesh,
                                                                     col_key_0,
                                                                     cell_key_0,
                                                                     col_key_1)
                                    for cell_key_1 in nhbr_keys:
                                        if cell_key_1 is not None:
                                            cell_1 = col_1.cells[cell_key_1]
                                            if cell_1.is_lf:
                                                # Get nhbr-cell info
                                                [th0_1, th1_1] = cell_1.pos[:]
                                                [dth_1]        = [th1_1 - th0_1]
                                                [nth_1]        = cell_1.ndofs[:]
                                                
                                                [_, _, _, _, thb_1, _] = qd.quad_xyth(nnodes_th = nth_1)
                                                
                                                # Get solution values in cell 1
                                                # F refers to the face of _0, so
                                                # it"s opposite for _1
                                                if F == 0:
                                                    uh_1 = uh_proj.cols[col_key_1].cells[cell_key_1].vals[0,:,:]
                                                elif F == 1:
                                                    uh_1 = uh_proj.cols[col_key_1].cells[cell_key_1].vals[:,0,:]
                                                elif F == 2:
                                                    uh_1 = uh_proj.cols[col_key_1].cells[cell_key_1].vals[-1,:,:]
                                                else: # F == 3
                                                    uh_1 = uh_proj.cols[col_key_1].cells[cell_key_1].vals[:,-1,:]
                                                
                                                # Get integration interval, quadrature info
                                                if dth_0 <= dth_1:
                                                    [th0_G, th1_G] = [th0_0, th1_0]
                                                    [dth_G]        = [dth_0]
                                                else:
                                                    [th0_G, th1_G] = [th0_1, th1_1]
                                                    [dth_G]        = [dth_1]
                                                if nth_0 >= nth_1:
                                                    [nth_H] = [nth_0]
                                                else:
                                                    [nth_H] = [nth_1]
                                                [_, _, _, _, thb_H, wth_H] = qd.quad_xyth(nnodes_th = nth_H)
                                                wth_H = wth_H.reshape([1, nth_H])
                                                thf_G = proj.push_forward(th0_G, th1_G, thb_H)
                                                thb_G_0 = proj.pull_back(th0_0, th1_0, thf_G)
                                                thb_G_1 = proj.pull_back(th0_1, th1_1, thf_G)
                                                
                                                xsi_0_mtx = np.zeros([nth_0, nth_H])
                                                for aa in range(0, nth_0):
                                                    for aa_p in range(0, nth_H):
                                                        xsi_0_mtx[aa, aa_p] = qd.lag_eval(thb_0, aa, thb_G_0[aa_p])
                                                        
                                                xsi_1_mtx = np.zeros([nth_1, nth_H])
                                                for rr in range(0, nth_1):
                                                    for aa_p in range(0, nth_H):
                                                        xsi_1_mtx[rr, aa_p] = qd.lag_eval(thb_1, rr, thb_G_1[aa_p])
                                                        
                                                if (F%2 == 0):
                                                    # Project uh_0, uh_1 to the same
                                                    # quadrature and integrate jump
                                                    uh_0_proj = np.zeros([ny_B, nth_H])
                                                    for jj in range(0, ny_0):
                                                        for aa in range(0, nth_0):
                                                            for jj_p in range(0, ny_B):
                                                                for aa_p in range(0, nth_H):
                                                                    uh_0_proj[jj_p, aa_p] += uh_0[jj, aa] * psi_0_mtx[jj, jj_p] * xsi_0_mtx[aa, aa_p]
                                                    uh_1_proj = np.zeros([ny_B, nth_H])
                                                    for qq in range(0, ny_1):
                                                        for rr in range(0, nth_1):
                                                            for jj_p in range(0, ny_B):
                                                                for aa_p in range(0, nth_H):
                                                                    uh_1_proj[jj_p, aa_p] += uh_1[qq, rr] * psi_1_mtx[qq, jj_p] * xsi_1_mtx[rr, aa_p]
                                                    col_jump += (dy_A * dth_G / 4.) * np.sum(wy_B * wth_H * (uh_0_proj - uh_1_proj)**2)
                                                else: # (F%2 == 1)
                                                    uh_0_proj = np.zeros([nx_B, nth_H])
                                                    for ii in range(0, nx_0):
                                                        for aa in range(0, nth_0):
                                                            for ii_p in range(0, nx_B):
                                                                for aa_p in range(0, nth_H):
                                                                    uh_0_proj[ii_p, aa_p] += uh_0[ii, aa] * phi_0_mtx[ii, ii_p] * xsi_0_mtx[aa, aa_p]
                                                    uh_1_proj = np.zeros([nx_B, nth_H])
                                                    for pp in range(0, nx_1):
                                                        for rr in range(0, nth_1):
                                                            for ii_p in range(0, nx_B):
                                                                for aa_p in range(0, nth_H):
                                                                    uh_1_proj[ii_p, aa_p] += uh_1[pp, rr] * phi_1_mtx[pp, ii_p] * xsi_1_mtx[rr, aa_p]
                                                    col_jump += (dx_A * dth_G / 4.) * np.sum(wx_B * wth_H * (uh_0_proj - uh_1_proj)**2)
                                                    
            col_err = np.sqrt((1. / (perim * 2. * np.pi)) * col_jump)
            col_max_err = max(col_max_err, col_err)
            
            if kwargs["ref_col"]:
                err_ind.cols[col_key_0].err = col_err
                
    # Determine hp-steering
    if kwargs["ref_col"]:
        err_ind.col_max_err = col_max_err
        col_ref_thrsh = col_ref_tol * col_max_err
        
        for col_key, col in col_items:
            if col.is_lf:
                col_err = err_ind.cols[col_key].err
                #p = col_err / col_max_err
                #to_ref = rng.choice([True, False], size = 1, p = (p, 1 - p))[0]
                if col_err >= col_ref_thrsh: # Does this one need to be refined?
                # Refine pseudo-randomly
                #if to_ref:
                    err_ind.avg_col_ref_err += col_err
                    col_nref += 1
                    if err_ind.cols[col_key].ref_form == "hp": # Does the form of refinement need to be chosen?
                        err_ind.cols[col_key].ref_form = hp_steer_col(mesh, uh_proj, col_key)
                else: # Needn't be refined
                    err_ind.cols[col_key].ref_form = None
    if kwargs["ref_col"]:
        err_ind.avg_col_ref_err /= col_nref
        
    return err_ind

def col_jump_err_old(mesh, uh_proj, **kwargs):
    
    default_kwargs = {"ref_col"      : True,
                      "col_ref_form" : "hp",
                      "col_ref_kind" : "spt",
                      "col_ref_tol"  : 0.85,
                      "ref_cell"      : False,
                      "cell_ref_form" : None,
                      "cell_ref_kind" : None,
                      "cell_ref_tol"  : None}
    kwargs = {**default_kwargs, **kwargs}
    
    err_ind = Error_Indicator(mesh, **kwargs)
    col_nref = 0
    
    col_items = sorted(mesh.cols.items())
    
    # Track maximum error(s) to calculate hp-steering only where needed
    col_max_err  = 0.
    col_ref_tol  = kwargs["col_ref_tol"]
    
    # Begin by angularly-integrating each column
    col_intg_ths = {}
    col_items = sorted(mesh.cols.items())
    for col_key, col in col_items:
        if col.is_lf:                                    
            col_intg_ths[col_key] = intg_col_bdry_th(mesh, uh_proj, col_key)
            
    # Once we have angularly-integrated along spatial faces of each col, we
    # spatially-integrate the jumps
    for col_key_0, col_0 in col_items:
        if col_0.is_lf:
            # Column information for quadrature
            [x0_0, y0_0, x1_0, y1_0] = col_0.pos[:]
            [dx_0, dy_0]             = [x1_0 - x0_0, y1_0 - y0_0]
            [ndof_x_0, ndof_y_0]     = col_0.ndofs[:]
            
            [xxb_0, wx_0, yyb_0, wy_0, _, _] = qd.quad_xyth(nnodes_x = ndof_x_0,
                                                            nnodes_y = ndof_y_0)
            
            xx1_0 = proj.push_forward(x0_0, x1_0, xxb_0)
            yyf_0 = proj.push_forward(y0_0, y1_0, yyb_0)

            # Loop through faces to calculate error
            col_err = 0.
            for F in range(0, 4):
                nhbr_keys = list(set(col_0.nhbr_keys[F]))
                
                col_intg_th_0 = col_intg_ths[col_key_0][F]
                
                for col_key_1 in nhbr_keys:
                    if col_key_1 is not None:
                        col_1 = mesh.cols[col_key_1]
                        if col_1.is_lf:
                            # Column information for quadrature
                            [x0_1, y0_1, x1_1, y1_1] = col_1.pos[:]
                            [dx_1, dy_1]             = [x1_1 - x0_1, y1_1 - y0_1]
                            [ndof_x_1, ndof_y_1]     = col_1.ndofs[:]
                            
                            [xxb_1, wx_1, yyb_1, wy_1, _, _] = qd.quad_xyth(nnodes_x = ndof_x_1,
                                                                            nnodes_y = ndof_y_1)
                            
                            col_intg_th_1 = col_intg_ths[col_key_1][(F+2)%4]
                            
                            if (F%2 == 0):
                                # Integrate over whichever face is smaller
                                if dy_0 >= dy_1:
                                    dcoeff = dy_1 / 2
                                    
                                    # Use quadrature rule for whichever column
                                    # has more nodes
                                    if (ndof_y_0 >= ndof_y_1):
                                        # Project col_0
                                        yyf_0_1 = proj.push_forward(y0_1, y1_1, yyb_0)
                                        yyb_0_1_0 = proj.pull_back(y0_0, y1_0, yyf_0_1)
                                        psi_jjp_matrix = np.zeros([ndof_y_0, ndof_y_0])
                                        for jj in range(0, ndof_y_0):
                                            for jj_p in range(0, ndof_y_0):
                                                psi_jjp_matrix[jj, jj_p] = \
                                                    qd.lag_eval(yyb_0, jj, yyb_0_1_0[jj_p])
                                                
                                        col_intg_th_0_0 = np.zeros(ndof_y_0)
                                        for jj_p in range(0, ndof_y_0):
                                            for jj in range(0, ndof_y_0):
                                                col_intg_th_0_0[jj_p] += col_intg_th_0[jj] * psi_jjp_matrix[jj, jj_p]
                                                
                                        # Project col_1
                                        psi_qjp_matrix = np.zeros([ndof_y_1, ndof_y_0])
                                        for qq in range(0, ndof_y_1):
                                            for jj_p in range(0, ndof_y_0):
                                                psi_qjp_matrix[qq, jj_p] = \
                                                    qd.lag_eval(yyb_1, qq, yyb_0[jj_p])
                                                
                                        col_intg_th_1_0 = np.zeros(ndof_y_0)
                                        for jj_p in range(0, ndof_y_0):
                                            for qq in range(0, ndof_y_1):
                                                col_intg_th_1_0[jj_p] += col_intg_th_1[qq] * psi_qjp_matrix[qq, jj_p]
                                                
                                        col_err += dcoeff * np.sum(wy_0 * (col_intg_th_0_0 - col_intg_th_1_0)**2)
                                                
                                    else: # ndof_y_0 < ndof_y_1
                                        # Project col_0
                                        yyf_1 = proj.push_forward(y0_1, y1_1, yyb_1)
                                        yyb_1_0 = proj.pull_back(y0_0, y1_0, yyf_1)
                                        psi_jqp_matrix = np.zeros([ndof_y_0, ndof_y_1])
                                        for jj in range(0, ndof_y_0):
                                            for qq_p in range(0, ndof_y_1):
                                                psi_jqp_matrix[jj, qq_p] = \
                                                    qd.lag_eval(yyb_0, jj, yyb_1_0[qq_p])
                                                                                                
                                        col_intg_th_0_1 = np.zeros(ndof_y_1)
                                        for qq_p in range(0, ndof_y_1):
                                            for jj in range(0, ndof_y_0):
                                                col_intg_th_0_1[qq_p] += col_intg_th_0[jj] * psi_jqp_matrix[jj, qq_p]
                                                
                                        col_err += dcoeff * np.sum(wy_1 * (col_intg_th_0_1 - col_intg_th_1)**2)
                                        
                                else: # dy_0 < dy_1:
                                    dcoeff = dy_0 / 2
                                    
                                    # Use quadrature rule for whichever column
                                    # has more nodes
                                    if (ndof_y_0 >= ndof_y_1):
                                        # Perform projection for col_1
                                        yyf_0 = proj.push_forward(y0_0, y1_0, yyb_0)
                                        yyb_0_1 = proj.pull_back(y0_1, y1_1, yyf_0)
                                        psi_qjp_matrix = np.zeros([ndof_y_1, ndof_y_0])
                                        for qq in range(0, ndof_y_1):
                                            for jj_p in range(0, ndof_y_0):
                                                psi_qjp_matrix[qq, jj_p] = \
                                                    qd.lag_eval(yyb_1, qq, yyb_0_1[jj_p])
                                                                                                
                                        col_intg_th_1_0 = np.zeros(ndof_y_0)
                                        for jj_p in range(0, ndof_y_0):
                                            for qq in range(0, ndof_y_1):
                                                col_intg_th_1_0[jj_p] += col_intg_th_1[qq] * psi_qjp_matrix[qq, jj_p]
                                                
                                        col_err += dcoeff * np.sum(wy_0 * (col_intg_th_0 - col_intg_th_1_0)**2)
                                        
                                    else: # ndof_y_0 < ndof_y_1
                                        # Perform projection for col_0
                                        psi_jqp_matrix = np.zeros([ndof_y_0, ndof_y_1])
                                        for jj in range(0, ndof_y_0):
                                            for qq_p in range(0, ndof_y_1):
                                                psi_jqp_matrix[jj, qq_p] = \
                                                    qd.lag_eval(yyb_0, jj, yyb_1[qq_p])
                                                
                                        col_intg_th_0_1 = np.zeros(ndof_y_1)
                                        for qq_p in range(0, ndof_y_1):
                                            for jj in range(0, ndof_y_0):
                                                col_intg_th_0_1[qq_p] += col_intg_th_0[jj] * psi_jqp_matrix[jj, qq_p]
                                                
                                        # Perform projection for col_1
                                        yyf_1_0 = proj.push_forward(y0_0, y1_0, yyb_1)
                                        yyb_1_0_1 = proj.pull_back(y0_1, y1_1, yyf_1_0)
                                        psi_qqp_matrix = np.zeros([ndof_y_1, ndof_y_1])
                                        for qq in range(0, ndof_y_1):
                                            for qq_p in range(0, ndof_y_1):
                                                psi_qqp_matrix[qq, qq_p] = \
                                                    qd.lag_eval(yyb_1, qq, yyb_1_0_1[qq_p])
                                                
                                        col_intg_th_1_1 = np.zeros(ndof_y_1)
                                        for qq_p in range(0, ndof_y_1):
                                            for qq in range(0, ndof_y_1):
                                                col_intg_th_1_1[qq_p] += col_intg_th_1[qq] * psi_qqp_matrix[qq, qq_p]
                                                
                                        col_err += dcoeff * np.sum(wy_1 * (col_intg_th_0_1 - col_intg_th_1_1)**2)
                                        
                            else: # (F%2) == 1
                                # Integrate over whichever face is smaller
                                if dx_0 >= dx_1:
                                    dcoeff = dx_1 / 2
                                    
                                    # Use quadrature rule for whichever column
                                    # has more nodes
                                    if (ndof_x_0 >= ndof_x_1):
                                        # Project col_0
                                        xxf_0_1 = proj.push_forward(y0_1, y1_1, xxb_0)
                                        xxb_0_1_0 = proj.pull_back(y0_0, y1_0, xxf_0_1)
                                        phi_iip_matrix = np.zeros([ndof_x_0, ndof_x_0])
                                        for ii in range(0, ndof_x_0):
                                            for ii_p in range(0, ndof_x_0):
                                                phi_iip_matrix[ii, ii_p] = \
                                                    qd.lag_eval(xxb_0, ii, xxb_0_1_0[ii_p])
                                                
                                        col_intg_th_0_0 = np.zeros(ndof_x_0)
                                        for ii_p in range(0, ndof_x_0):
                                            for ii in range(0, ndof_x_0):
                                                col_intg_th_0_0[ii_p] += col_intg_th_0[ii] * phi_iip_matrix[ii, ii_p]
                                                
                                        # Project col_1
                                        phi_pip_matrix = np.zeros([ndof_x_1, ndof_x_0])
                                        for pp in range(0, ndof_x_1):
                                            for ii_p in range(0, ndof_x_0):
                                                phi_pip_matrix[pp, ii_p] = \
                                                    qd.lag_eval(xxb_1, pp, xxb_0[ii_p])
                                                
                                        col_intg_th_1_0 = np.zeros(ndof_x_0)
                                        for ii_p in range(0, ndof_x_0):
                                            for pp in range(0, ndof_x_1):
                                                col_intg_th_1_0[ii_p] += col_intg_th_1[pp] * phi_pip_matrix[pp, ii_p]
                                                
                                        col_err += dcoeff * np.sum(wx_0 * (col_intg_th_0_0 - col_intg_th_1_0)**2)
                                                
                                    else: # ndof_x_0 < ndof_x_1
                                        # Project col_0
                                        xxf_1 = proj.push_forward(y0_1, y1_1, xxb_1)
                                        xxb_1_0 = proj.pull_back(y0_0, y1_0, xxf_1)
                                        phi_ipp_matrix = np.zeros([ndof_x_0, ndof_x_1])
                                        for ii in range(0, ndof_x_0):
                                            for pp_p in range(0, ndof_x_1):
                                                phi_ipp_matrix[ii, pp_p] = \
                                                    qd.lag_eval(xxb_0, ii, xxb_1_0[pp_p])
                                                                                                
                                        col_intg_th_0_1 = np.zeros(ndof_x_1)
                                        for pp_p in range(0, ndof_x_1):
                                            for ii in range(0, ndof_x_0):
                                                col_intg_th_0_1[pp_p] += col_intg_th_0[ii] * phi_ipp_matrix[ii, pp_p]
                                                
                                        col_err += dcoeff * np.sum(wx_1 * (col_intg_th_0_1 - col_intg_th_1)**2)
                                        
                                else: # dx_0 < dx_1:
                                    dcoeff = dx_0 / 2
                                    
                                    # Use quadrature rule for whichever column
                                    # has more nodes
                                    if (ndof_x_0 >= ndof_x_1):
                                        # Perform projection for col_1
                                        xxf_0 = proj.push_forward(y0_0, y1_0, xxb_0)
                                        xxb_0_1 = proj.pull_back(y0_1, y1_1, xxf_0)
                                        phi_pip_matrix = np.zeros([ndof_x_1, ndof_x_0])
                                        for pp in range(0, ndof_x_1):
                                            for ii_p in range(0, ndof_x_0):
                                                phi_pip_matrix[pp, ii_p] = \
                                                    qd.lag_eval(xxb_1, pp, xxb_0_1[ii_p])
                                                                                                
                                        col_intg_th_1_0 = np.zeros(ndof_x_0)
                                        for ii_p in range(0, ndof_x_0):
                                            for pp in range(0, ndof_x_1):
                                                col_intg_th_1_0[ii_p] += col_intg_th_1[pp] * phi_pip_matrix[pp, ii_p]
                                                
                                        col_err += dcoeff * np.sum(wx_0 * (col_intg_th_0 - col_intg_th_1_0)**2)
                                        
                                    else: # ndof_x_0 < ndof_x_1
                                        # Perform projection for col_0
                                        phi_ipp_matrix = np.zeros([ndof_x_0, ndof_x_1])
                                        for ii in range(0, ndof_x_0):
                                            for pp_p in range(0, ndof_x_1):
                                                phi_ipp_matrix[ii, pp_p] = \
                                                    qd.lag_eval(xxb_0, ii, xxb_1[pp_p])
                                                
                                        col_intg_th_0_1 = np.zeros(ndof_x_1)
                                        for pp_p in range(0, ndof_x_1):
                                            for ii in range(0, ndof_x_0):
                                                col_intg_th_0_1[pp_p] += col_intg_th_0[ii] * phi_ipp_matrix[ii, pp_p]
                                                
                                        # Perform projection for col_1
                                        xxf_1_0 = proj.push_forward(y0_0, y1_0, xxb_1)
                                        xxb_1_0_1 = proj.pull_back(y0_1, y1_1, xxf_1_0)
                                        phi_ppp_matrix = np.zeros([ndof_x_1, ndof_x_1])
                                        for pp in range(0, ndof_x_1):
                                            for pp_p in range(0, ndof_x_1):
                                                phi_ppp_matrix[pp, pp_p] = \
                                                    qd.lag_eval(xxb_1, pp, xxb_1_0_1[pp_p])
                                                
                                        col_intg_th_1_1 = np.zeros(ndof_x_1)
                                        for pp_p in range(0, ndof_x_1):
                                            for pp in range(0, ndof_x_1):
                                                col_intg_th_1_1[pp_p] += col_intg_th_1[pp] * phi_ppp_matrix[pp, pp_p]
                                                
                                        col_err += dcoeff * np.sum(wx_1 * ((1. / (2. * np.pi)) * (col_intg_th_0_1 - col_intg_th_1_1))**2)
                                        
            perim = 2. * dx_0 + 2. * dy_0
            col_err = np.sqrt((1. / perim) * col_err)
            col_max_err = max(col_max_err, col_err)
            
            if kwargs["ref_col"]:
                err_ind.cols[col_key_0].err = col_err
                
    # Weight to be relative error, determine hp-steering
    if kwargs["ref_col"]:
        err_ind.col_max_err = col_max_err
        col_ref_thrsh = col_ref_tol * col_max_err
        
        for col_key, col in col_items:
            if col.is_lf:
                col_err = err_ind.cols[col_key].err
                if col_err >= col_ref_thrsh: # Does this one need to be refined?
                    err_ind.avg_col_ref_err += col_err
                    col_nref += 1
                    if err_ind.cols[col_key].ref_form == "hp": # Does the form of refinement need to be chosen?
                        err_ind.cols[col_key].ref_form = hp_steer_col(mesh, uh_proj, col_key)
                else: # Needn't be refined
                    err_ind.cols[col_key].ref_form = None
    #if kwargs["ref_col"]:
    #    err_ind.avg_col_ref_err /= col_nref
        
    return err_ind

def intg_col_bdry_th(mesh, uh_proj, col_key):
    """
    Integrate the spatial boundary of a column with respect to theta.
    """

    proj_col = uh_proj.cols[col_key]
    if uh_proj_col.is_lf:
        [ndof_x, ndof_y] = uh_proj_col.ndofs[:]
        
        uh_proj_cell_items = sorted(uh_proj_col.cells.items())
        
        # Store the theta integral along each face
        # F = 0 = Right, proceed CCW
        col_intg_th = [np.zeros([ndof_y]), np.zeros([ndof_x]),
                       np.zeros([ndof_y]), np.zeros([ndof_x])]

        for cell_key, cell in uh_proj_cell_items:
            if cell.is_lf:
                [th0, thf] = cell.pos[:]
                dth        = thf - th0
                [ndof_th]  = cell.ndofs[:]

                [_, _, _, _, _, w_th] = qd.quad_xyth(nnodes_th = ndof_th)
                
        
                for F in range(0, 4):
                    if (F%2 == 0):
                        if (F == 0):
                            x_idx = ndof_x - 1
                        elif (F == 2):
                            x_idx = 0

                        for jj in range(0, ndof_y):
                            col_intg_th[F][jj] += \
                                (dth / 2.) * np.sum(w_th * cell.vals[x_idx, jj, :])
                                 
                    else:
                        if (F == 1):
                            y_idx = ndof_y - 1
                        elif (F == 3):
                            y_idx = 0

                        for ii in range(0, ndof_x):
                            col_intg_th[F][ii] += \
                                (dth / 2.) * np.sum(w_th * cell.vals[ii, y_idx, :])
                                
        return col_intg_th
