"""
Regression tests for issue #17: Error_Indicator.error_analytic must resolve the
analytic solution when it integrates (u - u_hp)**2, and must not reuse the
normalization Intg[u**2] of an earlier call.
"""

# Standard Library Imports

# Third-Party Library Imports
import numpy as np
from scipy.integrate import nquad

# Local Library Imports
import consts
from dg.mesh import Mesh
from dg.projection import Projection, push_forward
from dg.quadrature import lag_eval, quad_xyth
from amr.error_indicator import Error_Indicator

# Relative Imports

## Manufactured solution of experiments/manufactured-solution, whose angular
## profile has a standard deviation of about 0.046 rad
Ls: list = [3., 2.]

def u(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> np.ndarray:
    [Lx, Ly] = Ls
    X = np.exp(-((1. / Lx) * (x - (Lx / 3.)))**2)
    Y = np.exp(-4. * (Ly - y) / Ly)
    Theta = np.exp(-((96. / (2. * consts.PI)) * (th - (7. * consts.PI / 5.)))**2)
    return X * Y * Theta

def coarse_mesh() -> Mesh:
    # One column with ndofs [5, 5, 3] and eight angular cells of width pi/4, as
    # in the initial angular mesh of the manufactured-solution experiment
    mesh: Mesh = Mesh(Ls = Ls, pbcs = [False, False], ndofs = [5, 5, 3],
                      has_th = True)
    for _ in range(0, 3):
        mesh.ref_mesh(kind = "ang", form = "h")
    return mesh

def reference_error(proj: Projection, nnodes_xy: int, nnodes_th: int) -> float:
    # Relative L2 error with a fixed, fine quadrature rule on every cell
    [Lx, Ly] = Ls
    [intg_u2, _] = nquad(lambda x, y, th: u(x, y, th)**2,
                         [[0, Lx], [0, Ly], [0, 2. * consts.PI]])

    err: float = 0.
    for col_key, col in sorted(proj.cols.items()):
        [x0, y0, x1, y1] = col.pos[:]
        [nx, ny] = col.ndofs[:]
        [xxb, _, yyb, _, _, _] = quad_xyth(nnodes_x = nx, nnodes_y = ny)
        [xxh, wx, yyh, wy, _, _] = quad_xyth(nnodes_x = nnodes_xy,
                                             nnodes_y = nnodes_xy)
        phi = np.array([[lag_eval(xxb, ii, xx) for xx in xxh] for ii in range(0, nx)])
        psi = np.array([[lag_eval(yyb, jj, yy) for yy in yyh] for jj in range(0, ny)])
        xxf = push_forward(x0, x1, xxh)[:, None, None]
        yyf = push_forward(y0, y1, yyh)[None, :, None]
        for cell_key, cell in sorted(col.cells.items()):
            [th0, th1] = cell.pos[:]
            [nth] = cell.ndofs[:]
            [_, _, _, _, thb, _] = quad_xyth(nnodes_th = nth)
            [_, _, _, _, thh, wth] = quad_xyth(nnodes_th = nnodes_th)
            xsi = np.array([[lag_eval(thb, aa, th) for th in thh] for aa in range(0, nth)])
            uh = np.einsum("ija,ip,jq,ar->pqr", cell.vals, phi, psi, xsi)
            thf = push_forward(th0, th1, thh)[None, None, :]
            ww = wx[:, None, None] * wy[None, :, None] * wth[None, None, :]
            err += ((x1 - x0) * (y1 - y0) * (th1 - th0) / 8.) \
                * np.sum(ww * (u(xxf, yyf, thf) - uh)**2)
    return np.sqrt(err / intg_u2)

def test_error_analytic_zero_projection():
    # For u_hp = 0 the relative L2 error is exactly 1. With n + 3 = 6 angular
    # quadrature nodes per pi/4-wide cell the narrow peak is under-resolved.
    proj: Projection = Projection(coarse_mesh())
    err_ind: Error_Indicator = Error_Indicator(proj, ref_kind = "ang",
                                               ref_form = "h", ref_tol = [0., 0.])
    err_ind.error_analytic(u)

    assert(np.abs(err_ind.error - 1.) < 1.e-2)

def test_error_analytic_matches_fine_quadrature():
    # Interpolant of u on the coarse mesh, compared with a fine fixed rule
    proj: Projection = Projection(coarse_mesh(), u)
    err_ref: float = reference_error(proj, nnodes_xy = 16, nnodes_th = 40)

    err_ind: Error_Indicator = Error_Indicator(proj, ref_kind = "ang",
                                               ref_form = "h", ref_tol = [0., 0.])
    err_ind.error_analytic(u)

    assert(np.abs(err_ind.error - err_ref) < 1.e-2 * err_ref)

def test_error_analytic_normalization_per_call():
    # Polynomial solutions, integrated exactly by n + 3 nodes, so that only the
    # normalization Intg[u**2] can differ between the two calls
    def v(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> np.ndarray:
        return (1. + x + y) * (1. + th)
    def w(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> np.ndarray:
        return 2. * v(x, y, th)

    mesh: Mesh = Mesh(Ls = Ls, pbcs = [False, False], ndofs = [3, 3, 3],
                      has_th = True)
    proj: Projection = Projection(mesh)

    for anl_sol in [v, w]:
        err_ind: Error_Indicator = Error_Indicator(proj, ref_kind = "ang",
                                                   ref_form = "h", ref_tol = [0., 0.])
        err_ind.error_analytic(anl_sol)

        assert(np.abs(err_ind.error - 1.) < 1.e-6)
