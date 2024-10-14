# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports
import consts
from dg.quadrature import lag_eval, quad_xyth
from dg.projection import push_forward

# Relative Imports
from ..error_indicator_column import Error_Indicator_Column
from ..error_indicator_column.error_indicator_cell import Error_Indicator_Cell

def error_random(self, rng: np.random._generator.Generator = None) -> None:
    ## Random error in each column and cell
    
    ## Set RNG
    if rng is None:
        rng: np.random._generator.Generator = np.random.default_rng()
        
    ## Store maximum errors to calculate hp-steering only where needed
    col_max_err: float  = -consts.INF
    cell_max_err: float = -consts.INF
    
    ## Store the info needed for error_indicator
    cols: dict = {}
    mesh_err: float = 0.

    # Calculate the errors
    col_items: list = sorted(self.proj.mesh.cols.items())
    for col_key, col in col_items:
        assert(col.is_lf)
        
        # Store the info needed for error_indicator_columns
        col_err: float = 0.
        cells: dict = {}

        # Loop through cells to calculate error
        cell_items: list = sorted(col.cells.items())
        for cell_key, cell in cell_items:
            assert(cell.is_lf)

            cell_err: float = rng.random()

            col_err += cell_err
            mesh_err += cell_err
            
            cells[cell_key] = Error_Indicator_Cell(cell_err) # sqrt at the end to avoid sqrt then square
            cell_max_err: float = max(cell_max_err, cell_err)
                
        cols[col_key] = Error_Indicator_Column(col_err, cells)
        col_max_err: float = max(col_max_err, col_err)
    
    self.cols: dict = cols
    self.col_max_error: float = col_max_err
    self.cell_max_error: float = cell_max_err
    self.error: float = mesh_err

    ## Calculate if cols/cells need to be refined, and calculate hp-steering
    ang_ref_thrsh: float = self.ang_ref_tol * self.cell_max_error
    spt_ref_thrsh: float = self.spt_ref_tol * self.col_max_error
    for col_key, col in col_items:
        assert(col.is_lf)

        if self.ref_kind in ["spt", "all"]:
            if self.cols[col_key].error >= spt_ref_thrsh: # Does this one need to be refined?
                self.cols[col_key].do_ref = True
                if self.ref_form == "hp": # Does the form of refinement need to be chosen?
                    self.cols[col_key].ref_form = self.col_hp_steer(col_key)
                else:
                    self.cols[col_key].ref_form = self.ref_form
            else: # Needn't be refined
                self.cols[col_key].do_ref = False
        
        if self.ref_kind in ["ang", "all"]:
            cell_items: list = sorted(col.cells.items())
            for cell_key, cell in cell_items:
                assert(cell.is_lf)
                
                if self.cols[col_key].cells[cell_key].error >= ang_ref_thrsh: # Does this one need to be refined?
                    self.cols[col_key].cells[cell_key].do_ref = True
                    if self.ref_form == "hp": # Does the form of refinement need to be chosen?
                        self.cols[col_key].cells[cell_key].ref_form = \
                            self.cell_hp_steer(col_key, cell_key)
                    else:
                        self.cols[col_key].cells[cell_key].ref_form = self.ref_form
                else: # Needn't be refined
                    self.cols[col_key].cells[cell_key].do_ref = False