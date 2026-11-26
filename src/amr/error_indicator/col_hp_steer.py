# Standard Library Imports

# Third-Party Library Imports
import numpy as np
from scipy.special import eval_legendre

# Local Library Imports
import dg.quadrature as qd
from dg.projection.projection_column import Projection_Column

# Relative Imports


def col_hp_steer(self, col_key: int) -> str:
    col: Projection_Column = self.proj.cols[col_key]
    assert(col.is_lf)
    
    [nx, ny] = col.ndofs[:]
    [xxb, wx, yyb, wy, _, _] = qd.quad_xyth(nnodes_x = nx,
                                            nnodes_y = ny)
    xxb: np.ndarray = xxb.reshape([1, nx])
    wx: np.ndarray  =  wx.reshape([nx, 1])
    yyb: np.ndarray = yyb.reshape([1, ny])
    wy: np.ndarray  =  wy.reshape([1, ny])
    
    # uh_hat is the numerical solution integrated in angle
    uh_hat: np.ndarray = self.proj.col_intg_th(col_key)
    
    wx_wy_uh_hat: np.ndarray = wx * wy * uh_hat
    
    # The n-point LGL rule is exact only up to degree 2n - 3, so it gives
    # sum_i w_i L_{n-1}(x_i)^2 = 2 / (n - 1) instead of 2 / (2n - 1) for the
    # top mode. Normalizing the top index by (n - 1) / 2 instead of
    # (2n - 1) / 2 gives the exact Legendre coefficients of the nodal
    # interpolant of uh_hat used in SI Eq. (6b).
    
    # Calculate a_p^K,x, a_nx^K,x
    L_nxm = eval_legendre(nx - 1, xxb)
    ndofs_y: np.ndarray = np.arange(0, ny).reshape([1, ny])
    L_jn: np.ndarray = eval_legendre(ndofs_y.transpose(), yyb)
    
    nrm_y: np.ndarray = (2. * ndofs_y + 1.) / 2.
    nrm_y[0, ny - 1] = (ny - 1.) / 2.
    a_nxj: np.ndarray = (nx - 1.) / 2. * nrm_y \
        * (L_nxm @ wx_wy_uh_hat @ L_jn.transpose())
    
    ax_nx_sq: int = 0
    for jj in range(0, ny):
        ax_nx_sq += (a_nxj[0, jj])**2 * (2. / (2. * jj + 1.))
    
    # Calculate a_q^K,y, a_ny^K,y
    ndofs_x: np.ndarray = np.arange(0, nx).reshape([nx, 1])
    L_im: np.ndarray = eval_legendre(ndofs_x, xxb)
    L_nyn: np.ndarray = eval_legendre(ny - 1, yyb)
    
    nrm_x: np.ndarray = (2. * ndofs_x + 1.) / 2.
    nrm_x[nx - 1, 0] = (nx - 1.) / 2.
    a_iny: np.ndarray = nrm_x * (ny - 1.) / 2. \
        * (L_im @ wx_wy_uh_hat @ L_nyn.transpose())
    
    ay_ny_sq: int = 0
    for ii in range(0, nx):
        ay_ny_sq += (a_iny[ii, 0])**2 * (2. / (2. * ii + 1.))
    term_0: float = np.log((2. * nx - 1.) / (2 * ax_nx_sq)) / (2. * np.log(nx - 1.))
    term_1: float = np.log((2. * ny - 1.) / (2 * ay_ny_sq)) / (2. * np.log(ny - 1.))
    lp: float = 0.5 * (term_0 + term_1)

    lhs: float = lp - 0.5
    rhs: float = 0.5 * (nx + ny)
    if lhs >= rhs:
        ref_form: str = "p"
    else:
        ref_form: str = "h"
        
    return ref_form