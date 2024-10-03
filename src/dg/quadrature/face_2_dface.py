# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports

# Relative Imports
from .quad_xya import quad_xya
from .gl_proj import gl_proj

def face_2_dface(tablist, deg_prnt, deg_chld):
    """
    Create data structure responsible for face-to-double face communication for
    the mesh.
    
    tablist[0] - key to transfer matrix
    tablist[1] - Array of transfer matrix of size 
                 2*deg_chld X deg_prnt
    tablist[2] - Array of transfer matrix of size
                 deg_prnt X 2*deg_chld
    deg_prnt - Number of DOFs for the psarent face
    deg_chld - Number of DOFs for the child face
    """

    deg_sml = np.amin([deg_prnt, deg_chld])
    deg_lrg = np.amax([deg_prnt, deg_chld])
    deg_idx = [deg_sml, deg_lrg]

    if not tablist:
        tablist_upd = [dict(), dict()]
        tablist_upd[0][str(deg_idx)] = np.array([[1], [1]])
        tablist_upd[1][str(deg_idx)] = np.array([[0.5, 0.5]])
        [tab_f2df, tab_df2f, tablist_upd] = face_2_dface(tablist_upd,
                                                         deg_prnt, deg_chld)
    else:

        tablist_upd = tablist
        if str(deg_idx) not in tablist_upd[0].keys():
            if deg_chld == 1:
                mod_dist = np.eye(2)
                mod_coll = np.eye(2)
            else:
                mod_dist = np.zeros([2*deg_chld, 2*deg_chld - 1])
                mod_dist[0:deg_chld, 0:deg_chld] = np.eye(deg_chld)
                mod_dist[deg_chld:2*deg_chld, deg_chld-1:2*deg_chld-1] = \
                    np.eye(deg_chld)

                mod_coll = np.zeros([2*deg_chld - 1, 2*deg_chld])
                mod_coll[0:deg_chld-1, 0:deg_chld-1] = np.eye(deg_chld - 1)
                mod_coll[deg_chld-1, deg_chld-1:deg_chld+1] = \
                    np.asarray([0.5, 0.5])
                mod_coll[deg_chld:2*deg_chld - 1, deg_chld+1:2*deg_chld] = \
                    np.eye(deg_chld - 1)

            [nodes_prnt, _, nodes_chld, _, _, _] = \
                quad_xya(deg_prnt, deg_chld, 1)
            nodes_prnt = np.asarray(nodes_prnt)
            nodes_chld = np.asarray(nodes_chld)
            nodes_chld_2nm1 = np.union1d( (nodes_chld - 1.) / 2.,
                                          (nodes_chld + 1.) / 2. )

            if np.shape(nodes_chld_2nm1)[0] >= np.shape(nodes_prnt)[0]:
                [M_f2df, M_df2f] = gl_proj(nodes_chld_2nm1, nodes_prnt, True)
            else:
                [M_df2f, M_f2df] = gl_proj(nodes_prnt, nodes_chld_2nm1, True)
                
            tablist_upd[0][str(deg_idx)] = mod_dist @ M_f2df
            tablist_upd[1][str(deg_idx)] = M_df2f @ mod_coll

        tab_f2df = tablist_upd[0][str(deg_idx)]
        tab_df2f = tablist_upd[1][str(deg_idx)]

    return [tab_f2df, tab_df2f, tablist_upd]

    
