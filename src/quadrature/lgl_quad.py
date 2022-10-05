import numpy as np

def lgl_quad(order):
    '''
    Calculate Legendre-Gauss-Lobatto nodes, weights, and Vandermonde matrix.
    The LGL nodes are the zeros of (1 - x**2) * P'_N(x) (the Nth Legendre 
    polynomial).

    Based on the code of Greg von Winckel (2004/04/17)
    '''

    # Initial guess: Chebyshev-Gauss-Lobatto nodes
    nodes = np.cos(np.pi * np.arange(0, order+1) / order)

    # Vandermonde matrix
    vand = np.zeros([order+1, order+1])
    vand[:, 0] = 1
    # Compute P_n using the recursion relation
    # Compute its first and second derivatives using Newton-Raphson
    nodes_prev = nodes + 1
    while np.amax(np.abs(nodes - nodes_prev)) > 2.3e-16:
        vand[:, 1] = nodes
        for k in range(1, order):
            vand[:, k+1] = ( (2 * k + 1) * nodes * vand[:, k]
                             - k * vand[:, k-1] ) / (k + 1)

        nodes_prev = nodes
        nodes = nodes_prev \
            - ( (nodes * vand[:, order] - vand[:, order-1]) 
                / ((order + 1) * vand[:, order]) )


    weights = 2. / (order * (order + 1) * vand[:, order]**2)
    
    return [nodes, weights, vand]
