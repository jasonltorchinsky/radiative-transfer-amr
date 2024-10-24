# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports
import consts
import dg.quadrature as qd

# Relative Imports
from ..error_indicator_column import Error_Indicator_Column
from ..error_indicator_column.error_indicator_cell import Error_Indicator_Cell

def error_angular_jump(self) -> None:
    # Angular jump error
    
    # Store maximum errors to calculate hp-steering only where needed
    col_max_err: float  = -consts.INF
    cell_max_err: float = -consts.INF
    
    # Store the info needed for error_indicator
    cols: dict = {}
    mesh_err: float = 0.
    
    # Begin by getting the radiation field at the top and bottom of each cell
    col_items: list = sorted(self.proj.cols.items())
    cell_tops: dict = {}
    cell_bots: dict = {}
    for col_key, col in col_items:
        assert(col.is_lf)

        [nx, ny] = col.ndofs[:]
        cell_items: list = sorted(col.cells.items())
        for cell_key, cell in cell_items:
            assert(cell.is_lf)
            
            [nth] = cell.ndofs[:]
            
            [_, _, _, _, thb, _] = qd.quad_xyth(nnodes_th = nth)
            
            cell_vals: np.ndarray = cell.vals[:,:,:]
            cell_top: np.ndarray = np.zeros([nx, ny])
            cell_bot: np.ndarray = np.zeros([nx, ny])
            
            # Pull-back top is 1, pull-back bot is -1
            for aa in range(0, nth):
                cell_top += cell_vals[:,:,aa] * qd.lag_eval(thb, aa, 1)
                cell_bot += cell_vals[:,:,aa] * qd.lag_eval(thb, aa, -1)
                
            cell_tops[(col_key, cell_key)] = cell_top[:,:]
            cell_bots[(col_key, cell_key)] = cell_bot[:,:]
                    
    # Once we have the value of the radiation field at the top and bottom of
    # each cell, we can calculate the jumps and integrate those spatially
    for col_key, col in col_items:
        assert(col.is_lf)

        # Column information for weighting
        [nx, ny] = col.ndofs[:]
        
        [_, wx, _, wy, _, _] = qd.quad_xyth(nnodes_x = nx, nnodes_y = ny)
        wx: np.ndarray = wx.reshape([nx, 1])
        wy: np.ndarray = wy.reshape([1, ny])

        # Store the info needed for error_indicator_columns
        cells: dict = {}
        
        # Cell_0 is self
        col_err: float = 0.
        cell_items: list = sorted(col.cells.items())
        for cell_key_0, cell_0 in cell_items:
            assert(cell_0.is_lf)
                
            # Jump with lower-neighbor
            cell_bot_0: np.ndarray   = cell_bots[(col_key, cell_key_0)]
            nhbr_low_key: int = cell_0.nhbr_keys[0]
            nhbr_low_top: np.ndarray = cell_tops[(col_key, nhbr_low_key)]
            cell_jump_low: np.ndarray = (cell_bot_0 - nhbr_low_top)**2
            
            # Jump with upper-neighbor
            cell_top_0: np.ndarray   = cell_tops[(col_key, cell_key_0)]
            nhbr_up_key: int = cell_0.nhbr_keys[1]
            nhbr_up_bot: np.ndarray  = cell_bots[(col_key, nhbr_up_key)]
            cell_jump_up: np.ndarray = (cell_top_0 - nhbr_up_bot)**2

            # For integral, we multiply by dA / 4.
            # For mean, we divide integral by 1 / dA.
            # Hence, just divide by 4.
            # The 0.5 arise from the fact that we take the mean of all
            # the jumps, and there are two jumps.
            cell_err: np.ndarray = (1. / 4.) * np.sum(wx * wy * 0.5 * (cell_jump_low + cell_jump_up))
            col_err += cell_err
            mesh_err += cell_err

            cells[cell_key_0] = Error_Indicator_Cell(np.sqrt(cell_err)) # sqrt at the end to avoid sqrt then square
            cell_max_err: float = max(cell_max_err, np.sqrt(cell_err))

        cols[col_key] = Error_Indicator_Column(np.sqrt(col_err), cells)
        col_max_err: float = max(col_max_err, np.sqrt(col_err))
        
    self.cols: dict = cols
    self.col_max_error: float = col_max_err
    self.cell_max_error: float = cell_max_err
    self.error: float = np.sqrt(mesh_err)
    self.error_to_resolve: float = 0.
        
    ## Calculate if cols/cells need to be refined, and calculate hp-steering
    ang_ref_thrsh: float = self.ang_ref_tol * self.cell_max_error
    spt_ref_thrsh: float = self.spt_ref_tol * self.col_max_error
    for col_key, col in col_items:
        assert(col.is_lf)

        if self.ref_kind in ["spt", "all"]:
            if self.cols[col_key].error >= spt_ref_thrsh: # Does this one need to be refined?
                self.cols[col_key].do_ref = True
                self.error_to_resolve += self.cols[col_key].error
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
                    self.error_to_resolve += self.cols[col_key].cells[cell_key].error
                    if self.ref_form == "hp": # Does the form of refinement need to be chosen?
                        self.cols[col_key].cells[cell_key].ref_form = \
                            self.cell_hp_steer(col_key, cell_key)
                    else:
                        self.cols[col_key].cells[cell_key].ref_form = self.ref_form
                else: # Needn't be refined
                    self.cols[col_key].cells[cell_key].do_ref = False