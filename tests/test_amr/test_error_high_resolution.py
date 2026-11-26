"""
Regression test for issue #16: the high-resolution reference mesh must be the
low-resolution mesh with every element's degree raised by exactly
ang_ref_offset (in angle) and spt_ref_offset (in space).
"""

# Standard Library Imports
import types

# Third-Party Library Imports
import pytest

# Local Library Imports
from amr.error_indicator.error import error_high_resolution
from dg.mesh import Mesh


class _MeshCaptured(Exception):
    pass


class _CaptureProblem:
    """ Stands in for rt.Problem: records the mesh passed to solve() and stops. """
    def __init__(self):
        self.mesh = None

    def solve(self, mesh, **kwargs):
        self.mesh = mesh
        raise _MeshCaptured()


def _non_uniform_mesh() -> Mesh:
    mesh: Mesh = Mesh(Ls = [2., 1.], pbcs = [False, False], ndofs = [3, 3, 3],
                      has_th = True)
    mesh.ref_mesh(kind = "ang", form = "h")
    mesh.ref_mesh(kind = "ang", form = "h")
    mesh.ref_mesh(kind = "spt", form = "h")

    col_keys: list = sorted(mesh.cols.keys())
    col_key: int = col_keys[0]
    cell_key: int = sorted(mesh.cols[col_key].cells.keys())[-1]
    mesh.ref_cell(col_key, cell_key, form = "p") # One cell has n_theta = 4
    mesh.ref_col(col_key, kind = "spt", form = "p") # One column has n_x = 4

    return mesh


@pytest.mark.parametrize("ang_ref_offset, spt_ref_offset", [(1, 0), (0, 1), (2, 2)])
def test_high_resolution_mesh_offset_is_uniform(ang_ref_offset, spt_ref_offset):
    mesh: Mesh = _non_uniform_mesh()
    err_ind = types.SimpleNamespace(proj = types.SimpleNamespace(mesh = mesh))
    problem: _CaptureProblem = _CaptureProblem()

    with pytest.raises(_MeshCaptured):
        error_high_resolution(err_ind, problem,
                              ang_ref_offset = ang_ref_offset,
                              spt_ref_offset = spt_ref_offset)
    mesh_hr: Mesh = problem.mesh

    assert(sorted(mesh_hr.cols.keys()) == sorted(mesh.cols.keys()))
    for col_key, col in mesh.cols.items():
        col_hr = mesh_hr.cols[col_key]
        assert(col_hr.ndofs == [col.ndofs[0] + spt_ref_offset,
                                col.ndofs[1] + spt_ref_offset])
        assert(sorted(col_hr.cells.keys()) == sorted(col.cells.keys()))
        for cell_key, cell in col.cells.items():
            assert(col_hr.cells[cell_key].ndofs == [cell.ndofs[0] + ang_ref_offset])
