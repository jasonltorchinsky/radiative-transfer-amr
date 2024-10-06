# Standard Library Imports
import os

# Third-Party Library Imports
import numpy as np

# Local Library Imports
from dg.mesh import Mesh
from dg.projection import Projection, from_file
from tools.dg.mesh import plot_mesh
from tools.dg.projection import plot_th, plot_xth, plot_yth, plot_xy, plot_xyth

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

    ## Plot the mesh
    file_name: str = "mesh.png"
    file_path: str = os.path.join(tmp_path, file_name)
    plot_mesh(mesh, file_path = file_path)

    ## Create a Projection
    def f(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> np.ndarray:
        return np.cos(th) * (x + y)
    proj: Projection = Projection(mesh, f)

    ## Write the projection to file and read from it
    proj_file_name: str = "proj.npy"
    proj_file_path: str = os.path.join(tmp_path, proj_file_name)

    mesh_file_name: str = "mesh.json"
    mesh_file_path: str = os.path.join(tmp_path, mesh_file_name)

    proj.to_file(proj_file_path, write_mesh = True, mesh_file_path = mesh_file_path)

    same_proj: Projection = from_file(mesh_file_path, proj_file_path)
    assert(proj == same_proj)

    ## Plot the projection
    file_name: str = "proj_th.png"
    file_path: str = os.path.join(tmp_path, file_name)
    plot_th(proj, file_path = file_path)

    #file_name: str = "proj_xth.png"
    #file_path: str = os.path.join(tmp_path, file_name)
    #plot_xth(proj, file_path = file_path, cmap = "bwr", scale = "diff")

    #file_name: str = "proj_yth.png"
    #file_path: str = os.path.join(tmp_path, file_name)
    #plot_yth(proj, file_path = file_path, cmap = "bwr", scale = "diff")

    #file_name: str = "proj_xy.png"
    #file_path: str = os.path.join(tmp_path, file_name)
    #plot_xy(proj, file_path = file_path, cmap = "hot", scale = "normal")

    #file_name: str = "proj_xyth.png"
    #file_path: str = os.path.join(tmp_path, file_name)
    #plot_xyth(proj, file_path = file_path, cmap = "bwr", scale = "diff")