# Standard Library Imports
import json

# Third-Party Library Imports

# Local Library Imports

def write_mesh(mesh, out_fname : str = "mesh.json"):
    """
    Converts a mesh to a dict and writes it our to a .json file.
    """

    mesh_dict = {}

    # Mesh properties
    mesh_dict["Ls"] = mesh.Ls
    mesh_dict["pbcs"] = mesh.pbcs
    mesh_dict["has_th"] = mesh.has_th
    mesh_dict["cols"] = {}

    # Copy each column
    for col_key, col in mesh.cols.items():
        col_dict = {}

        col_dict["pos"]       = col.pos
        col_dict["idx"]       = col.idx
        col_dict["lv"]        = col.lv
        col_dict["key"]       = col.key
        col_dict["is_lf"]     = col.is_lf
        col_dict["ndofs"]     = col.ndofs
        col_dict["nhbr_keys"] = col.nhbr_keys

        # Copy each cell
        for cell_key, cell in col.cells.items():
            cell_dict = {}

            cell_dict["pos"]       = cell.pos
            cell_dict["idx"]       = cell.idx
            cell_dict["lv"]        = cell.lv
            cell_dict["key"]       = cell.key
            cell_dict["is_lf"]     = cell.is_lf
            cell_dict["ndofs"]     = cell.ndofs
            cell_dict["quad"]      = cell.quad
            cell_dict["nhbr_keys"] = cell.nhbr_keys

            col_dict[cell_key] = cell_dict

        mesh_dict[col_key] = col_dict

    with open(out_fname, "w") as out_file:
        json.dump(mesh_dict, out_file)