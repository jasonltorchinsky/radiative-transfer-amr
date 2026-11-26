"""
Regression test for issue #12: Problem.solve must reject meshes outside the
assumptions of the discretization instead of returning a wrong solution.
"""

# Standard Library Imports

# Third-Party Library Imports
import numpy as np
import pytest

# Local Library Imports
from dg.mesh import Mesh
from rt import Problem


def constant_problem() -> Problem:
    # kappa = 1, sigma = 0, f = 2, inflow value 2 => exact solution u = 2
    return Problem(lambda x, y: 1.0, lambda x, y: 0.0,
                   lambda th, phi: 1.0 / (2.0 * np.pi),
                   lambda x, y, th: 2.0,
                   [lambda x, y, th: 2.0, [None, None, None]])


def make_mesh(ndofs: list, nref_ang: int) -> Mesh:
    mesh: Mesh = Mesh(Ls = [1.0, 1.0], pbcs = [False, False], ndofs = ndofs,
                      has_th = True)
    for _ in range(0, nref_ang):
        mesh.ref_mesh(kind = "ang", form = "h")
    mesh.ref_mesh(kind = "spt", form = "h")
    return mesh


@pytest.mark.parametrize("nref_ang", [0, 1])
def test_solve_rejects_quadrant_spanning_cells(nref_ang):
    mesh: Mesh = make_mesh([3, 3, 3], nref_ang)
    with pytest.raises(ValueError, match = "quadrant"):
        constant_problem().solve(mesh)


def test_solve_rejects_one_spatial_node():
    mesh: Mesh = make_mesh([1, 1, 2], 2)
    with pytest.raises(ValueError, match = "LGL nodes"):
        constant_problem().solve(mesh)


def test_solve_accepts_supported_mesh():
    mesh: Mesh = make_mesh([3, 3, 3], 2)
    [uh, _, _] = constant_problem().solve(mesh)
    if uh is not None: # Only the root rank holds the solution
        err: float = max(np.abs(cell.vals - 2.0).max()
                         for col in uh.cols.values()
                         for cell in col.cells.values())
        assert err < 1.e-8
