def push_forward(x0, xf, nodes):
    """
    Transforms nodes on [-1, 1] to a [x0, xf].
    """

    xx = x0 + (xf - x0) / 2.0 * (nodes + 1.0)

    return xx

def pull_back(x0, xf, nodes):
    """
    Transforms nodes on [x0, xf] to a [-1, 1].
    """

    xx = -1.0 +  2.0 / (xf - x0) * (nodes - x0)

    return xx
