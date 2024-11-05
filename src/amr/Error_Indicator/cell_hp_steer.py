# Standard Library Imports

# Third-Party Library Imports
import numpy as np
from scipy.special import eval_legendre

# Local Library Imports
import dg.quadrature as qd
from dg.projection.projection_column import Projection_Column
from dg.projection.projection_column.projection_cell import Projection_Cell

# Relative Imports


def cell_hp_steer(self, col_key: int, cell_key: int) -> str:
    col: Projection_Column = self.proj.cols[col_key]
    assert(col.is_lf)

    cell: Projection_Cell = col.cells[cell_key]
    assert(cell.is_lf)
    
    [nth]  = cell.ndofs[:]
    
    [_, _, _, _, thb, wth] = qd.quad_xyth(nnodes_th = nth)
    
    thb: np.ndarray = thb.reshape([1, nth])
    wth: np.ndarray = wth.reshape([1, nth])
    
    # uh_hat is the numerical solution integrated in space
    uh_hat: np.ndarray = self.proj.cell_intg_xy(col_key, cell_key).reshape([1, nth])
    
    # Calculate a_p^K,th, a_nx^K,x
    L_nthr: np.ndarray = eval_legendre(nth - 1, thb)
    a_nth: np.ndarray = (2. * nth - 1.) / 2. \
        * (wth * uh_hat) @ L_nthr.transpose()
    ath_nth_sq: float = (a_nth[0, 0])**2 * (2. / (2. * nth - 1.))
    
    lp: float = np.log((2. * nth - 1.) / (2 * ath_nth_sq)) \
        / (2. * np.log(nth - 1))
    
    lhs: float = lp - 0.5
    rhs: float = float(nth)
    if lhs >= rhs:
        ref_form: str = "p"
    else:
        ref_form: str = "h"
    
    return ref_form
