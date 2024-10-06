# Standard Library Imports

# Third-Party Library Imports
import json

# Local Library Imports

# Relative Imports

def to_file(self, file_path: str = "mesh.json") -> None:
    # Mesh properties
    mesh_dict: dict = {}
    mesh_dict["Ls"]     = self.Ls
    mesh_dict["pbcs"]   = self.pbcs
    mesh_dict["has_th"] = self.has_th
    mesh_dict["cols"]   = {}

    # Copy each column
    for col_key, col in self.cols.items():
        col_dict: dict = {}

        col_dict["pos"]       = col.pos
        col_dict["idx"]       = col.idx
        col_dict["lv"]        = col.lv
        col_dict["key"]       = col.key
        col_dict["is_lf"]     = col.is_lf
        col_dict["ndofs"]     = col.ndofs
        col_dict["nhbr_keys"] = col.nhbr_keys
        col_dict["cells"]     = {}

        # Copy each cell
        for cell_key, cell in col.cells.items():
            cell_dict: dict = {}

            cell_dict["pos"]       = cell.pos
            cell_dict["idx"]       = cell.idx
            cell_dict["lv"]        = cell.lv
            cell_dict["key"]       = cell.key
            cell_dict["is_lf"]     = cell.is_lf
            cell_dict["ndofs"]     = cell.ndofs
            cell_dict["quadrant"]  = cell.quadrant
            cell_dict["nhbr_keys"] = cell.nhbr_keys

            col_dict["cells"][cell_key] = cell_dict

        mesh_dict["cols"][col_key] = col_dict

    with open(file_path, "w") as out_file:
        json.dump(mesh_dict, out_file)