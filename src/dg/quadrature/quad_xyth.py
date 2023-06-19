from .lg_quad import lg_quad
from .lgr_quad import lgr_quad
from .lgl_quad import lgl_quad
from .uni_quad import uni_quad

from utils import print_msg

import numpy as np

lgl_nodes = {}
lgl_weights = {}

lg_nodes = {}
lg_weights = {}

lgr_nodes = {}
lgr_weights = {}

uni_nodes = {}
uni_weights = {}

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
    elif nnodes_x in lgl_nodes.keys():
        [nodes_x, weights_x] = [lgl_nodes[nnodes_x], lgl_weights[nnodes_x]]
    elif nnodes_x == 1:
        [nodes_x, weights_x] = [np.asarray([0]), np.asarray([2])]

        lgl_nodes[nnodes_x] = nodes_x
        lgl_weights[nnodes_x] = weights_x
    else:
        [nodes_x, weights_x] = lgl_quad(nnodes_x)
        nodes_x = np.flip(nodes_x)
        weights_x = np.flip(weights_x)
        
        lgl_nodes[nnodes_x] = nodes_x
        lgl_weights[nnodes_x] = weights_x

    if nnodes_y < 1:
        print_msg('ERROR: ATTEMPTED TO CALCULATE QUADRATURE WEIGHTS WITH NNODES_Y < 1.')
        quit()
    elif nnodes_y in lgl_nodes.keys():
        [nodes_y, weights_y] = [lgl_nodes[nnodes_y], lgl_weights[nnodes_y]]
    elif nnodes_y == 1:
        [nodes_y, weights_y] = [np.asarray([0]), np.asarray([2])]
        
        lgl_nodes[nnodes_y] = nodes_y
        lgl_weights[nnodes_y] = weights_y
    else:
        [nodes_y, weights_y] = lgl_quad(nnodes_y)
        nodes_y = np.flip(nodes_y)
        weights_y = np.flip(weights_y)
        
        lgl_nodes[nnodes_y] = nodes_y
        lgl_weights[nnodes_y] = weights_y

    if nnodes_th < 1:
        print_msg('ERROR: ATTEMPTED TO CALCULATE QUADRATURE WEIGHTS WITH NNODES_TH < 1.')
        quit()
    elif nnodes_th in lg_nodes.keys():
        [nodes_th, weights_th] = [lg_nodes[nnodes_th], lg_weights[nnodes_th]]
    elif nnodes_th == 1:
        [nodes_th, weights_th] = [np.asarray([0]), np.asarray([2])]
        
        lg_nodes[nnodes_th] = nodes_th
        lg_weights[nnodes_th] = weights_th
    else:
        [nodes_th, weights_th] = lg_quad(nnodes_th)
        nodes_th = np.flip(nodes_th)
        weights_th = np.flip(weights_th)
        
        lg_nodes[nnodes_th] = nodes_th
        lg_weights[nnodes_th] = weights_th

    return [nodes_x, weights_x, nodes_y, weights_y, nodes_th, weights_th]
