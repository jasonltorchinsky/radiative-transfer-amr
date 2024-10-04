# Standard Library Imports
import os

# Third-Party Library Imports
import numpy as np

# Local Library Imports
from dg.mesh import Mesh
from dg.projection import Projection
from tools.dg.mesh import plot_mesh
from tools.dg.projection import plot_xyth

# Relative Imports

def test_Projection(tmp_path):
    # Create a mesh
    Ls: list = [1., 2.]
    pbcs: list = [False, False]
    ndofs: list = [3, 3, 3]
    has_th: bool = True

    mesh: Mesh = Mesh(Ls, pbcs, ndofs, has_th)

    for _ in range(0, 4):
        mesh.ref_mesh(kind = "all", form = "hp")

    ## Write mesh to file and plot it
    file_name: str = "mesh.json"
    file_path: str = os.path.join(tmp_path, file_name)
    mesh.to_file(file_path)

    file_name: str = "mesh.png"
    file_path: str = os.path.join(tmp_path, file_name)
    plot_mesh(mesh, file_path = file_path)

    ## Create a Projection and plot it
    def f(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> np.ndarray:
        return np.cos(th) * (x + y)
    proj: Projection = Projection(mesh, f)

    file_name: str = "proj.png"
    file_path: str = os.path.join(tmp_path, file_name)
    plot_xyth(proj, file_path = file_path, cmap = "bwr", scale = "diff")
