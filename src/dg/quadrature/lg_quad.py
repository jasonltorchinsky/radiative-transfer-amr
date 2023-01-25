import numpy as np

def lg_quad(nnodes, intv = [-1, 1]):
    '''
    Computes the nodes and weights needed for integration via Legendre-Gauss
    quadrature. Computes the nodes and weights on the interval intv = [a, b]
    with truncation order n.

    The order of the approximation is one greater than the number of nodes.

    Originally written by Greg von Winckel (2004/02/25)
    '''

    # Initial guess, we do the calculations on the interval [-1, 1] and
    # map it back to intv
    xu = np.linspace(-1, 1, nnodes)
    y = np.cos((2 * np.arange(0, nnodes) + 1) * np.pi / (2 * nnodes)) \
        + (0.27 / nnodes) * np.sin(np.pi * xu * (nnodes - 1) / (nnodes + 1))

    # Legendre-Gauss Vandermonde matrix
    vand = np.zeros([nnodes, nnodes + 1])
    vand[:, 0] = 1
    
    # Compute the zeros of the (n + 1) Legendre polynomial using
    # recursion relation and Newton-Raphson method
    y_prev = y + 1
    while np.amax(np.abs(y - y_prev)) > 2.3e-16:
        vand[:, 1] = y

        for k in range(1, nnodes):
            vand[:, k+1] = ( (2 * k + 1) * y * vand[:, k]
                             - k * vand[:, k-1] ) / (k + 1)

        # Derivative of last column of Vandermonde matrix
        vand_p = (nnodes + 1) * (vand[:, nnodes-1] - y * vand[:, nnodes]) \
            / (1 - y**2)

        y_prev = y
        y = y_prev - vand[:, nnodes] / vand_p

    # Compute the nodes and weights
    [a, b] = intv
    nodes = (a * (1 - y) + b * (1 + y)) / 2
    weights = ((nnodes + 1) / nnodes)**2 * (b - a) / ((1 - y**2) * vand_p**2)
    
    return [nodes, weights]
