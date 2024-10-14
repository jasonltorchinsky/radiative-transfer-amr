# Standard Library Imports
import json

# Third-Party Library Imports

# Local Library Imports
from dg.mesh.column import Column
from dg.mesh.column.cell import Cell
from dg.projection import Projection
from dg.projection import from_file as projection_from_file

# Relative Imports
from .class_Error_Indicator import Error_Indicator
from .error_indicator_column import Error_Indicator_Column
from .error_indicator_column.error_indicator_cell import Error_Indicator_Cell

def from_file(mesh_file_path: str = "mesh.json",
              projection_file_path: str = "projection.npy",
              err_ind_file_path: str = "err_ind.json") -> Error_Indicator:

    ## Load the projection from file
    proj: Projection = projection_from_file(mesh_file_path, projection_file_path)

    ## Load in error indicator dictionary
    with open(err_ind_file_path, "r") as in_file:
        err_ind_dict: dict = json.load(in_file)

    ## Initialize the error indicator
    err_ind: Error_Indicator = Error_Indicator(proj = proj,
                                               ref_kind = err_ind_dict["ref_kind"],
                                               ref_form = err_ind_dict["ref_form"],
                                               ref_tol = [err_ind_dict["ang_ref_tol"], err_ind_dict["spt_ref_tol"]])

    err_ind.error: float = err_ind_dict["mesh_error"]
    err_ind.col_max_error: float = err_ind_dict["col_max_error"]
    err_ind.cell_max_error: float = err_ind_dict["cell_max_error"]

    ## Set up the error indicator columns, cells
    err_ind_cols: dict = {}
    
    col_keys: list = sorted(err_ind_dict["cols"].keys())

    for col_key in col_keys:
        col_key_int: int = int(col_key)
        col: Column = err_ind.proj.mesh.cols[col_key_int]
        assert(col.is_lf)

        col_dict: dict = err_ind_dict["cols"][col_key]

        cells: dict = {}
        cell_keys: list = sorted(col_dict["cells"].keys())
        for cell_key in cell_keys:
            cell_key_int: int = int(cell_key)
            cell: Cell = col.cells[cell_key_int]
            assert(cell.is_lf)

            cell_dict: dict = col_dict["cells"][cell_key]
            cells[cell_key_int]: Error_Indicator_Cell = Error_Indicator_Cell(cell_dict["error"],
                                                                             cell_dict["ref_form"],
                                                                             cell_dict["do_ref"])
            
        err_ind_cols[col_key_int]: Error_Indicator_Column = Error_Indicator_Column(col_dict["error"],
                                                                                   cells,
                                                                                   col_dict["ref_form"],
                                                                                   col_dict["do_ref"])

    err_ind.cols: dict = err_ind_cols

    return err_ind