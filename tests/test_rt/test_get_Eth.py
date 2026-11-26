"""
Regression test for issue #10: the upwind inflow term at a non-conforming
angular face must use the receiving cell's own angular quadrature, the rule of
its volume and outflow terms, so that a constant state is preserved.
"""

# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import pytest
from mpi4py import MPI

# Local Library Imports
import consts
from dg.mesh import Mesh
from dg.projection import push_forward
from dg.quadrature import quad_xyth
from rt import Problem
from rt.get_Eth import get_Eth, Theta_F_func

# Relative Imports

def make_mesh(form: str, ndof_th: int) -> Mesh:
    """
    2 x 2 columns with 4 angular cells each. The first cell of the first column
    is refined in angle (form = "h" or "p"), so that the angular faces between
    this column and its spatial neighbors are non-conforming.
    """
    mesh: Mesh = Mesh(Ls = [2., 1.], pbcs = [False, False],
                      ndofs = [3, 3, ndof_th], has_th = True)
    for _ in range(0, 2):
        mesh.ref_mesh(kind = "ang", form = "h")
    mesh.ref_mesh(kind = "spt", form = "h")

    col_key: int = sorted(mesh.cols.keys())[0]
    cell_key: int = sorted(mesh.cols[col_key].cells.keys())[0]
    mesh.ref_cell(col_key, cell_key, form = form)

    return mesh

@pytest.mark.parametrize("ndof_th", [3, 4])
@pytest.mark.parametrize("form", ["h", "p"])
def test_get_Eth_constant_trace(form: str, ndof_th: int):
    """
    For a constant neighbor trace, the inflow factor summed over the neighboring
    cells must equal the outflow factor w_r * Theta_F(th_r) of the receiving
    cell, i.e., the same angular quadrature as its own trace.
    """
    mesh: Mesh = make_mesh(form, ndof_th)

    nfaces: int = 0
    for col_key_0, col_0 in sorted(mesh.cols.items()):
        for F in range(0, 4):
            nhbr_col_keys: list = list(set(col_0.nhbr_keys[F]) - {None})
            for col_key_1 in nhbr_col_keys:
                for cell_key_0, cell_0 in sorted(col_0.cells.items()):
                    [th0_0, th1_0] = cell_0.pos[:]
                    [ndof_th_0] = cell_0.ndofs[:]
                    [_, _, _, _, thb_0, wth_0] = quad_xyth(nnodes_th = ndof_th_0)
                    thf_0: np.ndarray = push_forward(th0_0, th1_0, thb_0)
                    own: np.ndarray = wth_0 * Theta_F_func(thf_0, F)

                    inflow: np.ndarray = np.zeros([ndof_th_0])
                    for cell_key_1 in mesh.nhbr_cells_in_nhbr_col(col_key_0,
                                                                  cell_key_0,
                                                                  col_key_1):
                        if cell_key_1 is not None:
                            E_th: np.ndarray = get_Eth(mesh,
                                                       col_key_0, cell_key_0,
                                                       col_key_1, cell_key_1,
                                                       F)
                            inflow += np.sum(E_th, axis = 0)

                    assert(np.allclose(inflow, own, rtol = 0., atol = 1.e-14))
                    nfaces += 1

    assert(nfaces > 0)

@pytest.mark.parametrize("scatter", [False, True])
@pytest.mark.parametrize("form", ["h", "p"])
def test_constant_state_non_conforming_angular_mesh(form: str, scatter: bool):
    """
    With f = (kappa - sigma) * c, isotropic scattering and inflow data c, the
    exact solution is u = c. The discrete solution must be c to round-off.
    """
    mesh: Mesh = make_mesh(form, 3)

    c: float = 2.
    def kappa(x, y):
        return 1. + 0.5 * x
    if scatter:
        def sigma(x, y):
            return 0.5 * kappa(x, y)
    else:
        def sigma(x, y):
            return 0. * x
    def Phi(th, phi):
        return 1. / (2. * np.pi) + 0. * th
    def f(x, y, th):
        return (kappa(x, y) - sigma(x, y)) * c
    def bcs(x, y, th):
        return c + 0. * x
    problem: Problem = Problem(kappa, sigma, Phi, f, [bcs, [None, None, None]])

    [uh, _, _] = problem.solve(mesh, ksp_type = "gmres", pc_type = "lu")

    if MPI.COMM_WORLD.Get_rank() == consts.COMM_ROOT:
        err: float = 0.
        for col in uh.cols.values():
            for cell in col.cells.values():
                err = max(err, np.amax(np.abs(cell.vals - c)))
        assert(err < 1.e-12)
