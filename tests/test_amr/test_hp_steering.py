"""
Regression tests for the hp-steering formulas:

- issue #7: the spatial-angular steering criterion (paper Eq. 16) uses the
  averages of the marked error indicators, not their sums;
- issue #8: the angular hp-steering criterion (paper Eq. 18b) uses the squared
  top Legendre coefficient |gamma_f|^2;
- issue #9: the spatial hp-steering criterion (SI Eq. 6b) uses the exact top
  Legendre coefficients, although the LGL rule is not exact for the top mode.
"""

# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import pytest
from scipy.special import eval_legendre

# Local Library Imports
import consts
from amr.error_indicator import Error_Indicator
from amr.error_indicator.error_indicator_column import Error_Indicator_Column
from amr.error_indicator.error_indicator_column.error_indicator_cell import Error_Indicator_Cell
from dg.mesh import Mesh
from dg.projection import Projection, pull_back


def make_mesh(n: int) -> Mesh:
    # One column [0, 2]^2 (area 4) with 4 angular cells
    mesh: Mesh = Mesh(Ls = [2., 2.], pbcs = [False, False], ndofs = [n, n, n],
                      has_th = True)
    mesh.ref_mesh(kind = "ang", form = "h")
    mesh.ref_mesh(kind = "ang", form = "h")
    return mesh


def expected_form(lp: float, n: float) -> str:
    return "p" if lp - 0.5 >= n else "h"


## Issue #8: angular hp-steering, paper Eq. (18b)
@pytest.mark.parametrize("nth", [3, 4, 5])
def test_cell_hp_steer_follows_eq_18b(nth: int):
    mesh: Mesh = make_mesh(nth)
    cell_key: int = sorted(mesh.cols[0].cells.keys())[0]
    [th0, th1] = mesh.cols[0].cells[cell_key].pos[:]

    # lp of Eq. (18b) as a function of gamma_f; pick gamma_f on both sides of
    # the threshold lp - 1/2 = nth, including values where the old extra
    # factor 2 / (2 nth - 1) flipped the decision
    def lp_eq_18b(gamma: float) -> float:
        return np.log((2. * nth - 1.) / (2. * gamma**2)) / (2. * np.log(nth - 1.))
    gamma_thrsh: float = np.sqrt((2. * nth - 1.) / 2.) * (nth - 1.)**(-(nth + 0.5))
    for ratio in [0.3, 0.7, 0.9, 1.1, 1.5, 2.5, 4.]:
        gamma: float = ratio * gamma_thrsh

        # u = c P_{nth-1}(th_ref) on each cell, so g(th) = 4 c P_{nth-1} on
        # this cell and gamma_f = 4 c exactly
        c: float = gamma / 4.
        def u(x, y, th):
            return c * eval_legendre(nth - 1, pull_back(th0, th1, th))
        proj: Projection = Projection(mesh, u)
        err_ind: Error_Indicator = Error_Indicator(proj, ref_kind = "ang",
                                                   ref_form = "hp")

        assert err_ind.cell_hp_steer(0, cell_key) == \
            expected_form(lp_eq_18b(gamma), nth), (nth, ratio)


## Issue #9: spatial hp-steering, SI Eq. (6b)
@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("shape", ["sum", "product"])
def test_col_hp_steer_follows_si_eq_6b(n: int, shape: str):
    mesh: Mesh = make_mesh(n)
    p: int = n - 1

    # g = sum_K int_K u dth = 2 pi u for u independent of th. With
    # x_ref = x - 1, y_ref = y - 1:
    # - "sum": g = c (P_p(x_ref) + P_p(y_ref)), so gamma_p0 = gamma_0p = c and
    #   (gamma^x)^2 = (gamma^y)^2 = 2 c^2;
    # - "product": g = c P_p(x_ref) P_p(y_ref), so gamma_pp = c and
    #   (gamma^x)^2 = (gamma^y)^2 = 2 c^2 / (2 p + 1), which also checks the
    #   normalization of the top index in the inner sums.
    if shape == "sum":
        def g_ref(xr, yr):
            return eval_legendre(p, xr) + eval_legendre(p, yr)
        nrm: float = 2.
    else:
        def g_ref(xr, yr):
            return eval_legendre(p, xr) * eval_legendre(p, yr)
        nrm: float = 2. / (2. * p + 1.)

    def lp_si_eq_6b(c: float) -> float:
        term: float = np.log((2. * n - 1.) / (2. * nrm * c**2)) / (2. * np.log(n - 1.))
        return 0.5 * (term + term)
    c_thrsh: float = np.sqrt((2. * n - 1.) / (2. * nrm)) * (n - 1.)**(-(n + 0.5))
    for ratio in [0.2, 0.5, 0.9, 1.1, 1.5, 2.5, 6.]:
        c: float = ratio * c_thrsh
        def u(x, y, th):
            return c / (2. * consts.PI) * g_ref(x - 1., y - 1.) + 0. * th
        proj: Projection = Projection(mesh, u)
        err_ind: Error_Indicator = Error_Indicator(proj, ref_kind = "spt",
                                                   ref_form = "hp")

        assert err_ind.col_hp_steer(0) == expected_form(lp_si_eq_6b(c), n), \
            (n, shape, ratio)


## Issue #7: spatial-angular steering, paper Eq. (16)
def test_mean_error_to_resolve_is_average_over_marked_elements():
    # Spatial indicator: 4 columns, 2 marked; angular indicator: 4 columns of
    # 8 cells, all 32 marked (as with tau_s = 0)
    col_errs: list = [1.0, 0.9, 0.3, 0.2]
    err_ind_spt: Error_Indicator = Error_Indicator(None, ref_kind = "spt")
    err_ind_spt.cols = {}
    for col_key, err in enumerate(col_errs):
        cells: dict = {cell_key: Error_Indicator_Cell(0.5) for cell_key in range(8)}
        err_ind_spt.cols[col_key] = Error_Indicator_Column(err, cells,
                                                           do_ref = (err >= 0.8))
    err_ind_spt.error_to_resolve = 1.0 + 0.9

    rng: np.random.Generator = np.random.default_rng(0)
    cell_errs: np.ndarray = rng.uniform(0.05, 0.15, size = [4, 8])
    err_ind_ang: Error_Indicator = Error_Indicator(None, ref_kind = "ang")
    err_ind_ang.cols = {}
    for col_key in range(4):
        cells: dict = {cell_key: Error_Indicator_Cell(cell_errs[col_key, cell_key],
                                                      do_ref = True)
                       for cell_key in range(8)}
        err_ind_ang.cols[col_key] = Error_Indicator_Column(1.0, cells)
    err_ind_ang.error_to_resolve = float(np.sum(cell_errs))

    eta_x: float = err_ind_spt.mean_error_to_resolve()
    eta_s: float = err_ind_ang.mean_error_to_resolve()
    assert eta_x == pytest.approx(0.95)
    assert eta_s == pytest.approx(np.mean(cell_errs))

    # Eq. (16): p_x = mean(eta_x) / (mean(eta_x) + mean(eta_s)), not the
    # ratio of the sums
    p_x: float = eta_x / (eta_x + eta_s)
    assert p_x == pytest.approx(0.95 / (0.95 + np.mean(cell_errs)))
    assert p_x > 0.85
    p_x_sums: float = err_ind_spt.error_to_resolve \
        / (err_ind_spt.error_to_resolve + err_ind_ang.error_to_resolve)
    assert p_x_sums < 0.5


def test_mean_error_to_resolve_after_angular_jump():
    # The average agrees with the marked elements of a computed indicator
    mesh: Mesh = make_mesh(3)
    mesh.ref_mesh(kind = "spt", form = "h")
    def u(x, y, th):
        return np.exp(np.cos(th)) * (1. + x * y)
    proj: Projection = Projection(mesh, u)

    for [ref_kind, ref_tol] in [["ang", [0., consts.INF]],
                                ["spt", [consts.INF, 0.]],
                                ["all", [0.5, 0.5]]]:
        err_ind: Error_Indicator = Error_Indicator(proj, ref_kind = ref_kind,
                                                   ref_form = "h",
                                                   ref_tol = ref_tol)
        err_ind.error_angular_jump()

        errs: list = []
        for col in err_ind.cols.values():
            if col.do_ref:
                errs += [col.error]
            for cell in col.cells.values():
                if cell.do_ref:
                    errs += [cell.error]
        assert len(errs) > 0
        assert err_ind.mean_error_to_resolve() == pytest.approx(np.mean(errs)), ref_kind
