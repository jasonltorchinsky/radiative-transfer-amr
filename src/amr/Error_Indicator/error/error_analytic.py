# Standard Library Imports
from typing import Callable

# Third-Party Library Imports
import numpy as np
from scipy.integrate import nquad

# Local Library Imports
import consts
from dg.projection import Projection
import dg.quadrature as qd
from dg.projection import push_forward

# Relative Imports
from .error_indicator_column import Error_Indicator_Column
from .error_indicator_column.error_indicator_cell import Error_Indicator_Cell
from .hp_steer import hp_steer_col, hp_steer_cell

intg_u2: float = None

def error_analytic(self, anl_sol: Callable[[np.ndarray, np.ndarray, np.ndarray],
                                            np.ndarray]) -> None:
    # L2 error by cell and column (Sqrt[Sum[cell_err**2]] / Sqrt[Intg[u**2]])
    
    # Integrate square of analytic solution to weight the relative error
    if intg_u2 is None:
        [Lx, Ly] = self.proj.mesh.Ls[:]
        [intg_u2, _] = nquad(lambda x, y, th: (anl_sol(x, y, th))**2,
                             [[0, Lx], [0, Ly], [0, 2. * consts.PI]])
        
    # Store maximum errors to calculate hp-steering only where needed
    col_max_err: float  = -consts.INF
    cell_max_err: float = -consts.INF
    
    # Store the info needed for error_indicator
    cols: dict = {}

    # Calculate the errors
    col_items: list = sorted(self.proj.mesh.cols.items())
    for col_key, col in col_items:
        assert(col.is_lf)
        
        # Column information for quadrature
        [x0, y0, xf, yf] = col.pos[:]
        [dx, dy] = [xf - x0, yf - y0]
        [nx, ny] = col.ndofs[:]
        
        [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = nx,
                                                nnodes_y = ny)
        
        xxf: np.ndarray = push_forward(x0, xf, xxb).reshape(nx, 1, 1)
        wx: np.ndarray  = wx.reshape(nx, 1, 1)
        yyf: np.ndarray = push_forward(y0, yf, yyb).reshape(1, ny, 1)
        wy: np.ndarray  = wy.reshape(1, ny, 1)
        
        # Store the info needed for error_indicator_columns
        cells: dict = {}

        # Loop through cells to calculate error
        col_err: float = 0.
        cell_items: list = sorted(col.cells.items())
        for cell_key, cell in cell_items:
            assert(cell.is_lf)

            # Cell information for quadrature
            [th0, thf] = cell.pos[:]
            dth: float = thf - th0
            [nth]  = cell.ndofs[:]
            
            [_, _, _, _, thb, wth] = qd.quad_xyth(nnodes_th = nth)
            
            thf: np.ndarray = push_forward(th0, thf, thb).reshape(1, 1, nth)
            wth: np.ndarray = wth.reshape(1, 1, nth)
            
            # Calculate error
            uh_cell: np.ndarray = self.proj.cols[col_key].cells[cell_key].vals[:,:,:]
            u_cell: np.ndarray  = anl_sol(xxf, yyf, thf)
            
            cell_err: float = \
                np.sum((dx * dy * dth / 8.) * wx * wy * wth * (u_cell - uh_cell)**2) / intg_u2
            col_err += cell_err
            
            cells[cell_key] = Error_Indicator_Cell(np.sqrt(cell_err)) # sqrt at the end to avoid sqrt then square
            cell_max_err: float = max(cell_max_err, cell_err)
                
        cols[col_key] = Error_Indicator_Column(np.sqrt(col_err), cells)
        col_max_err: float = max(col_max_err, col_err)
    
    self.cols: dict = cols
    self.col_max_err: float = col_max_err
    self.cell_max_err: float = cell_max_err

    ## Calculate if cols/cells need to be refined, and calculate hp-steering
    ang_ref_thrsh: float = self.ang_ref_tol * self.cell_max_err
    spt_ref_thrsh: float = self.spt_ref_tol * self.col_max_err
    for col_key, col in col_items:
        assert(col.is_lf)

        if self.ref_kind in ["spt", "all"]:
            if self.cols[col_key].error >= spt_ref_thrsh: # Does this one need to be refined?
                self.cols[col_key].do_ref = True
                if self.ref_form == "hp": # Does the form of refinement need to be chosen?
                    self.cols[col_key].ref_form = self.hp_steer_col(col_key)
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
                            self.hp_steer_cell(col_key, cell_key)
                    else:
                        self.cols[col_key].cells[cell_key].ref_form = self.ref_form
                else: # Needn't be refined
                    self.cols[col_key].cells[cell_key].do_ref = False