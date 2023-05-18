import numpy as np

def lgr_quad(nnodes = 5):
    '''
    Calculate Legendre-Gauss-Radau nodes, weights, and Vandermonde matrix.
    The LGR nodes are the zeros of P_N + P'_N(x) (the Nth Legendre 
    polynomial).

    The order of approximation is one greater than the number of nodes.

    Based on the code of Greg von Winckel (2004/05/02).

    Reference: C. Canuto, M. Y. Hussaini, A. Quarteroni, T. A. Tang,
    "Spectral Methods in Fluid Dynamics," Section 2.3. Springer-Verlag 1987
    '''

    # Initial guess: Chebyshev-Gauss-Radau nodes
    nodes = -np.cos(2. * np.pi * np.arange(0, nnodes) / (2. * nnodes - 1))

    # Vandermonde matrix
    vand = np.zeros([nnodes, nnodes + 1])

    nodes_prev = np.copy(nodes[:]) + 1.
    while np.amax(np.abs(nodes - nodes_prev)) > 2.3e-16:
        vand[0,  :] = (-1.)**(np.arange(0, nnodes + 1))
        vand[1:, 0] = 1.
        vand[1:, 1] = nodes[1:]
        for kk in range(1, nnodes):
            vand[1:, kk + 1] = ( (2 * kk + 1) * nodes[1:] * vand[1:, kk]
                                - kk * vand[1:, kk - 1] ) / (kk + 1)

        nodes_prev = np.copy(nodes)
        nodes[1:] = nodes_prev[1:] \
            - ( ((1. - nodes_prev[1:]) / nnodes)
                * (vand[1:, nnodes - 1] + vand[1:, nnodes])
                / (vand[1:, nnodes - 1] - vand[1:, nnodes])
                )

    weights = np.zeros(nnodes)
    weights[0] = 2. / nnodes**2
    weights[1:] = (1. - nodes[1:]) / (nnodes * vand[1:, nnodes - 1])**2
    
    return [nodes, weights]
