# Standard Library Imports
import os

# Third-Party Library Imports

# Local Library Imports
import consts
from dg.mesh import Mesh, from_file
from tools.dg.mesh import plot_mesh

def test_Mesh(tmp_path):
    # Create a mesh
    Ls: list = [1., 2.]
    pbcs: list = [False, False]
    ndofs: list = [2, 3, 4]
    has_th: bool = True

    mesh: Mesh = Mesh(Ls, pbcs, ndofs, has_th)

    ## Check mesh.__str__()
    file_name: str = "mesh_str.txt"
    file_path: str = os.path.join(tmp_path, file_name)
    file = open(file_path, "w")
    file.write(mesh.__str__())
    file.close()

    ## Write mesh to and read mesh from file
    file_name: str = "mesh.json"
    file_path: str = os.path.join(tmp_path, file_name)
    mesh.to_file(file_path)

    same_mesh: Mesh = from_file(file_path)
    assert(mesh == same_mesh)

    ## Plot mesh
    file_name: str = "mesh.png"
    file_path: str = os.path.join(tmp_path, file_name)
    plot_mesh(mesh, file_path = file_path)

    ## Refine mesh and plot it again
    mesh.ref_col(col_key = 0, kind = "all", form = "h")
    mesh.ref_col(col_key = 1, kind = "all", form = "h")
    mesh.ref_col(col_key = 5, kind = "all", form = "hp")
    mesh.ref_col(col_key = 30, kind = "all", form = "hp")
    file_name: str = "mesh_ref.png"
    file_path: str = os.path.join(tmp_path, file_name)
    plot_mesh(mesh, file_path = file_path)
