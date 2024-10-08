# Standard Library Imports
import os

# Third-Party Library Imports

# Local Library Imports
from dg.projection import Projection
from amr.error_indicator import Error_Indicator
from tools.amr import plot_error_indicator
from tools.dg.mesh import plot_mesh
from tools.dg.projection import plot_xyth

# Relative Imports

def generate_plots(uh: Projection, err_ind: Error_Indicator, dir_path: str) -> None:
    ## Plot the mesh and the numeric solution
    file_name : str = "mesh.png"
    file_path: str = os.path.join(dir_path, file_name)
    plot_mesh(uh.mesh, file_path = file_path)

    ## Plot the mesh and the numeric solution
    file_name : str = "uh_xyth.png"
    file_path: str = os.path.join(dir_path, file_name)
    plot_xyth(uh, file_path = file_path)

    ## Refine the mesh and plot the error indicator
    file_name: str = "err_ind.png"
    file_path: str = os.path.join(dir_path, file_name)
    plot_error_indicator(err_ind, file_path, name = "Analytic Error")