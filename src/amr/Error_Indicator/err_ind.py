import numpy as np
import sys

sys.path.append('/quadrature')
import quadrature as qd
sys.path.append('/mesh')
from mesh import ji_mesh, tools

from .Projection import Projection

def err_ind(mesh, uh):

    if uh.has_a:
        uh_xy = uh.intg_a(mesh)
    else:
        uh_xy = uh
    # Assume is not normalized
    uh_xy.coeffs = (np.array(uh_xy.coeffs) / (2 * np.pi)).tolist()

    err = Projection(mesh = mesh, has_a = False)
    [_, tablist_f2f] = qd.face_2_face(dict(), 1, 1)
    [_, _, tablist_f2df] = qd.face_2_dface([], 1, 1)
    faces = ['r', 't', 'l', 'b']
    for key, is_lf in sorted(mesh.is_lf.items()):
        if is_lf:
            ijlv = mesh.ijlv[key]
            dof_x = mesh.dof_x[key]
            dof_y = mesh.dof_y[key]
            dof_a = mesh.dof_a[key]
            [x0, y0, x1, y1] = mesh.pos[key]
            dx = x1 - x0
            dy = y1 - y0
            
            st_idx = uh_xy.st_idxs[key]
            
            [_, weights_x, _, weights_y, _, _] = qd.quad_xya(dof_x, dof_y, dof_a)
            uh_xy_elt = uh_xy.coeffs[st_idx:st_idx + dof_x * dof_y]
            uh_xy_elt = np.asarray(uh_xy_elt).reshape(dof_x, dof_y, order = 'F')
            
            p_list = [[dof_x - 1], range(0, dof_x), [0], range(0, dof_x)]
            q_list = [range(0, dof_y), [dof_y - 1], range(0, dof_y), [0]]
            
            bdry_intg = 0
            for face_idx in range(0, 4):
                face = faces[face_idx]
                
                [flag, nhbr_1, nhbr_2] = ji_mesh.fnd_nhbr(mesh, ijlv, face)
                
                if flag == 'f0':
                    key_nhbr = ji_mesh.get_key(nhbr_1)
                    dof_x_nhbr = mesh.dof_x[key_nhbr]
                    dof_y_nhbr = mesh.dof_y[key_nhbr]
                    
                    st_idx_nhbr = uh_xy.st_idxs[key_nhbr]
                    uh_xy_elt_nhbr = \
                        uh_xy.coeffs[st_idx_nhbr:st_idx_nhbr + dof_x_nhbr * dof_y_nhbr]
                    uh_xy_elt_nhbr = \
                        np.asarray(uh_xy_elt_nhbr).reshape(dof_x_nhbr, dof_y_nhbr, order = 'F')
                    
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
                        
                    me = uh_xy_elt[p_list[face_idx], q_list[face_idx]]
                    nhbr = uh_xy_elt_nhbr[p_list_nhbr[face_idx],
                                          q_list_nhbr[face_idx]]
                    df = me.flatten(order = 'F') \
                        - tab_nhbr2me @ nhbr.flatten(order = 'F')
                    
                elif flag == 'cc':
                    key_nhbr_1 = ji_mesh.get_key(nhbr_1)
                    st_idx_nhbr_1 = uh_xy.st_idxs[key_nhbr_1]
                    
                    key_nhbr_2 = ji_mesh.get_key(nhbr_2)
                    st_idx_nhbr_2 = uh_xy.st_idxs[key_nhbr_2]
                    
                    dof_x_nhbr_1 = mesh.dof_x[key_nhbr_1]
                    dof_y_nhbr_1 = mesh.dof_y[key_nhbr_1]
                    
                    st_idx_nhbr_1 = uh_xy.st_idxs[key_nhbr_1]
                    uh_xy_elt_nhbr_1 = \
                        uh_xy.coeffs[st_idx_nhbr_1:st_idx_nhbr_1 + dof_x_nhbr_1 * dof_y_nhbr_1]
                    uh_xy_elt_nhbr_1 = \
                        np.asarray(uh_xy_elt_nhbr_1).reshape(dof_x_nhbr_1, dof_y_nhbr_1, order = 'F')

                    dof_x_nhbr_2 = mesh.dof_x[key_nhbr_2]
                    dof_y_nhbr_2 = mesh.dof_y[key_nhbr_2]
                    st_idx_nhbr_2 = uh_xy.st_idxs[key_nhbr_2]
                    uh_xy_elt_nhbr_2 = \
                        uh_xy.coeffs[st_idx_nhbr_2:st_idx_nhbr_2 + dof_x_nhbr_2 * dof_y_nhbr_2]
                    uh_xy_elt_nhbr_2 = \
                        np.asarray(uh_xy_elt_nhbr_2).reshape(dof_x_nhbr_2, dof_y_nhbr_2, order = 'F')
                    
                    p_list_nhbr_1 = [[0], range(0, dof_x_nhbr_1),
                                     [dof_x_nhbr_1 - 1], range(0, dof_x_nhbr_1)]
                    q_list_nhbr_1 = [range(0, dof_y_nhbr_1), [0],
                                     range(0, dof_y_nhbr_1), [dof_y_nhbr_1 - 1]]

                    p_list_nhbr_2 = [[0], range(0, dof_x_nhbr_2),
                                     [dof_x_nhbr_2 - 1], range(0, dof_x_nhbr_2)]
                    q_list_nhbr_2 = [range(0, dof_y_nhbr_2), [0],
                                     range(0, dof_y_nhbr_2), [dof_y_nhbr_2 - 1]]

                    # dof_x_nhbr_1 and dof_x_nhbr_2 should be the same, I believe?
                    
                    if (face == 'r' or face == 'l'):
                        [_, tab_nhbr12me, tablist_f2df] = \
                            qd.face_2_dface(tablist_f2df, dof_y, dof_y_nhbr_1)
                        [_, tab_nhbr22me, tablist_f2df] = \
                            qd.face_2_dface(tablist_f2df, dof_y, dof_y_nhbr_2)
                    elif (face == 't' or face == 'b'):
                        [_, tab_nhbr12me, tablist_f2df] = \
                            qd.face_2_dface(tablist_f2df, dof_x, dof_x_nhbr_1)
                        [_, tab_nhbr22me, tablist_f2df] = \
                            qd.face_2_dface(tablist_f2df, dof_x, dof_x_nhbr_2)
                        
                    me = uh_xy_elt[p_list[face_idx], q_list[face_idx]]
                    nhbr_1 = uh_xy_elt_nhbr_1[p_list_nhbr_1[face_idx],
                                              q_list_nhbr_1[face_idx]]
                    nhbr_2 = uh_xy_elt_nhbr_2[p_list_nhbr_2[face_idx],
                                              q_list_nhbr_2[face_idx]]
                    nhbr = np.concatenate((nhbr_1.flatten(order = 'F'),
                                           nhbr_2.flatten(order = 'F')))
                    # Something is messed up in here, but I'm not sure what it is
                    df = me.flatten(order = 'F') \
                        - tab_nhbr12me @ nhbr.flatten(order = 'F')
                    
                elif (flag == 'pm' or flag == 'pp'):
                    key_nhbr = ji_mesh.get_key(nhbr_1)
                    dof_x_nhbr = mesh.dof_x[key_nhbr]
                    dof_y_nhbr = mesh.dof_y[key_nhbr]
                    
                    st_idx_nhbr = uh_xy.st_idxs[key_nhbr]
                    
                    st_idx_nhbr = uh_xy.st_idxs[key_nhbr]
                    uh_xy_elt_nhbr = \
                        uh_xy.coeffs[st_idx_nhbr:st_idx_nhbr + dof_x_nhbr * dof_y_nhbr]
                    uh_xy_elt_nhbr = \
                        np.asarray(uh_xy_elt_nhbr).reshape(dof_x_nhbr, dof_y_nhbr, order = 'F')
                    
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
                        
                    me = uh_xy_elt[p_list[face_idx], q_list[face_idx]]
                    nhbr = uh_xy_elt_nhbr[p_list_nhbr[face_idx],
                                          q_list_nhbr[face_idx]]
                    
                    end = np.shape(tab_nhbr2me)[0]
                    if flag == 'pm':
                        df = me.flatten(order = 'F') \
                            - tab_nhbr2me[0:int(end/2), :] @ nhbr.flatten(order = 'F')
                    elif flag == 'pp':
                        df = me.flatten(order = 'F') \
                            - tab_nhbr2me[int(end/2):, :] @ nhbr.flatten(order = 'F')
                        
                df_abs = np.abs(df)
                if (face == 'r' or face == 'l'):
                    temp = np.sum(df_abs.flatten(order = 'F')**2 * weights_y) \
                        * (dy / 2)
                elif (face == 't' or face == 'b'):
                    temp = np.sum(df_abs.flatten(order = 'F')**2 * weights_x) \
                        * (dx / 2)
                    
                bdry_intg += temp
                    
            error = np.sqrt(bdry_intg / (2 * (dx + dy)))
            st_idx_err = err.st_idxs[key]
            # The line below sets the entry to be a 1X1 array, which needs to be fixed!
            err.coeffs[st_idx:st_idx + dof_x * dof_y] \
                = (error * np.ones([dof_x * dof_y])).tolist()
            
            
    return err
