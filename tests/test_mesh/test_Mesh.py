# Standard Library Imports
import os

# Third-Party Library Imports

# Local Library Imports
import consts
from mesh import Mesh, from_file

def test_Mesh(tmp_path):
    # Create a mesh
    Ls: list = [1., 2.]
    pbcs: list = [False, False]
    ndofs: list = [2, 3, 4]
    has_th: bool = True

    mesh: Mesh = Mesh(Ls, pbcs, ndofs, has_th)

    file_name: str = "mesh_str.txt"
    file_path: str = os.path.join(tmp_path, file_name)
    file = open(file_path, "w")
    file.write(mesh.__str__())
    file.close()

    file_name: str = "mesh.json"
    file_path: str = os.path.join(tmp_path, file_name)
    mesh.to_file(file_path)

    same_mesh: Mesh = from_file(file_path)
    assert(mesh == same_mesh)