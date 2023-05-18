import numpy as np

def lgl_quad(nnodes = 5):
    '''
    Calculate Legendre-Gauss-Lobatto nodes, weights, and Vandermonde matrix.
    The LGL nodes are the zeros of (1 - x**2) * P'_N(x) (the Nth Legendre 
    polynomial).

    The order of approximation is equal to the number of nodes.

    Based on the code of Greg von Winckel (2004/04/17).

    Reference: C. Canuto, M. Y. Hussaini, A. Quarteroni, T. A. Tang,
    "Spectral Methods in Fluid Dynamics," Section 2.3. Springer-Verlag 1987
    '''

    # Initial guess: Chebyshev-Gauss-Lobatto nodes
    nodes = np.cos(np.pi * np.arange(0, nnodes) / (nnodes - 1))

    # Vandermonde matrix
    vand = np.zeros([nnodes, nnodes])
    # Compute P_n using the recursion relation
    # Compute its first and second derivatives using Newton-Raphson
    nodes_prev = np.copy(nodes) + 1.
    while np.amax(np.abs(nodes - nodes_prev)) > 2.3e-16:
        vand[:, 0] = 1
        vand[:, 1] = nodes
        for kk in range(1, nnodes - 1):
            vand[:, kk + 1] = ( (2 * kk + 1) * nodes * vand[:, kk]
                                - kk * vand[:, kk - 1] ) / (kk + 1)

        nodes_prev = np.copy(nodes)
        nodes = nodes_prev \
            - ( (nodes * vand[:, nnodes - 1] - vand[:, nnodes - 2]) 
                / ((nnodes + 1) * vand[:, nnodes - 1]) )


    weights = 2. / ((nnodes - 1) * nnodes * vand[:, nnodes - 1]**2)
    
    return [nodes, weights]
