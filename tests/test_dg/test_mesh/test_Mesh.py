# Standard Library Imports
import os

# Third-Party Library Imports

# Local Library Imports
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

    ## Plot initial mesh
    file_name: str = "mesh_0.png"
    file_path: str = os.path.join(tmp_path, file_name)
    plot_mesh(mesh, file_path = file_path)

    ## Refine mesh and plot it again
    col_keys: list = [0, 0, 0, 1, 10, 48]
    ref_kinds: list = ["ang", "ang", "spt", "spt", "spt", "spt"]
    ref_forms: list = ["h", "h", "h", "h", "hp", "hp"]

    nrefs: int = 0
    
    for idx in range(0, len(col_keys)):
        nrefs += 1
        mesh.ref_col(col_key = col_keys[idx], kind = ref_kinds[idx], 
                     form = ref_forms[idx])

        file_name: str = "mesh_{}.png".format(nrefs)
        file_path: str = os.path.join(tmp_path, file_name)
        plot_mesh(mesh, file_path = file_path, show_p = True)

    col_key: int = 204
    cell_keys: list = [3, 7, 15]
    ref_forms: list = ["h", "hp", "hp"]

    for idx in range(0, len(cell_keys)):
        #breakpoint()
        nrefs += 1
        mesh.ref_cell(col_key = col_key, cell_key = cell_keys[idx], 
                     form = ref_forms[idx])

        file_name: str = "mesh_{}.png".format(nrefs)
        file_path: str = os.path.join(tmp_path, file_name)
        plot_mesh(mesh, file_path = file_path, show_p = True)
