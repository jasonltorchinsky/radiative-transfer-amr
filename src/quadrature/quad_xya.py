from .lgl_quad import lgl_quad
from .lg_quad import lg_quad

import numpy as np

def quad_xya(order_x, order_y, order_a):
    '''
    Returns quadrature points, weights for x-, y-, and angular (a-) directions.

    We use Gauss-Lobatto for x-, y-.
    We use Legendre-Gauss for a-.
    '''

    # Caution: No catches for non-positive orders?
    if order_x == 1:
        [nodes_x, weights_x] = [[0], [2]]
    else:
        [nodes_x, weights_x, _] = lgl_quad(order_x - 1)
        nodes_x = np.flip(nodes_x)

    if order_y == 1:
        [nodes_y, weights_y] = [[0], [2]]
    else:
        [nodes_y, weights_y, _] = lgl_quad(order_y - 1)
        nodes_y = np.flip(nodes_y)

    if order_a == 1:
        [nodes_a, weights_a] = [[0], [2]]
    else:
        [nodes_a, weights_a] = lg_quad(order_a, [-1, 1])
        nodes_a = np.flip(nodes_a)


    return [nodes_x, weights_x, nodes_y, weights_y, nodes_a, weights_a]
