# Standard Library Imports
import json

# Third-Party Library Imports
import numpy as np

# Local Library Imports

# Relative Imports

def to_file(self, file_path: str = "err_ind.json", **kwargs) -> None:
    default_kwargs: dict = {"write_mesh" : True,
                            "mesh_file_path" : "mesh.json",
                            "write_projection" : True,
                            "projection_file_path" : "projection.npy"}
    kwargs: dict = {**default_kwargs, **kwargs}

    if kwargs["write_projection"]:
        self.proj.to_file(kwargs["projection_file_path"], **kwargs)

    err_ind_dict: dict = {}
    err_ind_dict["ref_kind"] = self.ref_kind
    err_ind_dict["ref_form"] = self.ref_form
    err_ind_dict["ang_ref_tol"] = self.ang_ref_tol
    err_ind_dict["spt_ref_tol"] = self.spt_ref_tol
    err_ind_dict["mesh_error"] = self.error
    err_ind_dict["col_max_error"] = self.col_max_error
    err_ind_dict["cell_max_error"] = self.cell_max_error

    err_ind_cols: dict = {}
    col_items: list = sorted(self.proj.mesh.cols.items())

    for col_key, col in col_items:
        assert(col.is_lf)

        err_ind_col: dict = {}
        err_ind_col["do_ref"] = self.cols[col_key].do_ref
        err_ind_col["ref_form"] = self.cols[col_key].ref_form
        err_ind_col["error"] = self.cols[col_key].error

        err_ind_col_cells: dict = {}
        cell_items: list = sorted(self.proj.mesh.cols[col_key].cells.items())
        for cell_key, cell in cell_items:
            assert(cell.is_lf)

            err_ind_cell: dict = {}
            err_ind_cell["do_ref"] = self.cols[col_key].cells[cell_key].do_ref
            err_ind_cell["ref_form"] = self.cols[col_key].cells[cell_key].ref_form
            err_ind_cell["error"] = self.cols[col_key].cells[cell_key].error

            err_ind_col_cells[cell_key] = err_ind_cell

        err_ind_col["cells"] = err_ind_col_cells
        
        err_ind_cols[col_key] = err_ind_col

    err_ind_dict["cols"] = err_ind_cols
    
    with open(file_path, "w") as out_file:
        json.dump(err_ind_dict, out_file)