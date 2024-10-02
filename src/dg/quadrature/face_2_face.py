import numpy as np

from .quad_xya import quad_xya
from .gl_proj import gl_proj

def face_2_face(tablist, deg_in, deg_out):
    """
    Create data structure responsible for face-to-face communication for
    the mesh.
    
    tablist[0] - key to transfer matrix
    tablist[1] - Array of transfer matrix of size 
                 max(deg_in, deg_out) X max(deg_in, deg_out)
    tablist[2] - ""
    deg_in, deg_out - Input/output number of DOFs (# of rows/columns)
    """

    deg_sml = np.amin([deg_in, deg_out])
    deg_lrg = np.amax([deg_in, deg_out])
    deg_idx = [deg_sml, deg_lrg]
    
    if not tablist:
        tablist_upd = [dict(), dict()]
        tablist_upd[0][str(deg_idx)] = np.eye(1)
        tablist_upd[1][str(deg_idx)] = np.eye(1)
        [tab_f2f, tablist_upd] = face_2_face(tablist_upd, deg_in, deg_out)
    else:

        tablist_upd = tablist
        if str(deg_idx) not in tablist_upd[0].keys():
            [nodes_lrg, _, nodes_sml, _, _, _] = quad_xya(deg_lrg, deg_sml, 1)
            [A, Adag] = gl_proj(nodes_lrg, nodes_sml, True)
            tablist_upd[0][str(deg_idx)] = A
            tablist_upd[1][str(deg_idx)] = Adag
            
        if deg_in < deg_out:
            tab_f2f = tablist_upd[0][str(deg_idx)]
        else:
            tab_f2f = tablist_upd[1][str(deg_idx)]

    return [tab_f2f, tablist_upd]
