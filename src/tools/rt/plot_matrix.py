# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import matplotlib.pyplot as plt

# Local Library Imports
from dg.mesh import Mesh

def plot_matrix(matrix: np.ndarray, file_path: str = "out.png", **kwargs):
    default_kwargs: dict = {"title" : None,
                            "marker_size": 2.}
    kwargs: dict = {**default_kwargs, **kwargs}

    ## Visualize the matrix
    fig, ax = plt.subplots()
    
    ax.spy(matrix,
           marker     = "s",
           markersize = kwargs["marker_size"],
           color      = "k")
                    
    if kwargs["title"] is not None:
        ax.set_title(kwargs["title"])
    
    fig.set_size_inches(6.5, 6.5)
    plt.savefig(file_path, dpi = 300, bbox_inches = "tight")
    plt.close(fig)