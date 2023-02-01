from .lgl_quad import lgl_quad
from .lg_quad import lg_quad

from utils import print_msg

import numpy as np

def quad_xyth(nnodes_x = 1, nnodes_y = 1, nnodes_th = 1):
    '''
    Returns quadrature points, weights for spatial (x-, y-)
    and angular (th-) directions.

    We use Gauss-Lobatto for x-, y-.
    We use Legendre-Gauss for th-.
    '''

    # Caution: No catches for non-positive orders?
    if nnodes_x < 1:
        print_msg('ERROR: ATTEMPTED TO CALCULATE QUADRATURE WEIGHTS WITH NNODES_X < 1.')
        quit()
    elif nnodes_x == 1:
        [nodes_x, weights_x] = [[0], [2]]
    else:
        [nodes_x, weights_x] = lgl_quad(nnodes_x)
        nodes_x = np.flip(nodes_x)

    if nnodes_y < 1:
        print_msg('ERROR: ATTEMPTED TO CALCULATE QUADRATURE WEIGHTS WITH NNODES_Y < 1.')
        quit()
    elif nnodes_y == 1:
        [nodes_y, weights_y] = [[0], [2]]
    else:
        [nodes_y, weights_y] = lgl_quad(nnodes_y)
        nodes_y = np.flip(nodes_y)

    if nnodes_th < 1:
        print_msg('ERROR: ATTEMPTED TO CALCULATE QUADRATURE WEIGHTS WITH NNODES_TH < 1.')
        quit()
    elif nnodes_th == 1:
        [nodes_th, weights_th] = [[0], [2]]
    else:
        [nodes_th, weights_th] = lg_quad(nnodes_th)
        nodes_th = np.flip(nodes_th)


    return [nodes_x, weights_x, nodes_y, weights_y, nodes_th, weights_th]
