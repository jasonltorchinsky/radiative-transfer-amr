# Standard Library Imports

# Third-Party Library Imports
import json

# Local Library Imports
from .column import Column
from .column.cell import Cell

from .class_Mesh import Mesh

# Relative Imports

def from_file(file_path: str = "mesh.json") -> Mesh:
    # Load in mesh dictionary
    with open(file_path, "r") as in_file:
        mesh_dict: dict = json.load(in_file)

    # Make the mesh object, empty out its columns
    Ls: list     = mesh_dict["Ls"]
    pbcs: list   = mesh_dict["pbcs"]
    has_th: bool = mesh_dict["has_th"]
    mesh: Mesh   = Mesh(Ls = Ls, pbcs = pbcs, has_th = has_th)
    mesh.del_col(0)

    # Copy each Column to the mesh
    for col_key in mesh_dict["cols"].keys():
        col_dict: dict  = mesh_dict["cols"][col_key]

        col_pos: list   = col_dict["pos"]
        col_idx: list   = col_dict["idx"]
        col_lv: int     = col_dict["lv"]
        col_is_lf: bool = col_dict["is_lf"]
        col_ndofs: list = col_dict["ndofs"]
        col_nhbr_keys: list = col_dict["nhbr_keys"]

        col_cells: dict = {}

        # Copy each Cell to the mesh
        for cell_key in col_dict["cells"].keys():
            cell_key_int: int = int(cell_key)
            cell_dict: dict = col_dict["cells"][cell_key]

            cell_pos: list     = cell_dict["pos"]
            cell_idx: int      = cell_dict["idx"]
            cell_lv: int       = cell_dict["lv"]
            cell_is_lf: bool   = cell_dict["is_lf"]
            cell_ndofs: list   = cell_dict["ndofs"]
            cell_quadrant: int = cell_dict["quadrant"]
            cell_nhbr_keys: list = cell_dict["nhbr_keys"]

            col_cells[cell_key_int] = Cell(pos = cell_pos, idx = cell_idx,
                                           lv = cell_lv, is_lf = cell_is_lf, 
                                           ndofs = cell_ndofs,
                                           quadrant = cell_quadrant,
                                           nhbr_keys = cell_nhbr_keys)

        col: Column = Column(pos = col_pos, idx = col_idx, lv = col_lv,
                             is_lf = col_is_lf, ndofs = col_ndofs,
                             cells = col_cells, nhbr_keys = col_nhbr_keys)
        
        mesh.add_col(col)

    return mesh