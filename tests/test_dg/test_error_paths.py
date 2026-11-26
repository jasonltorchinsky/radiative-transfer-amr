"""
Regression test for issue #11: invalid refinement and quadrature requests must
raise an exception instead of calling sys.exit(0) / quit(), which reported
success and, under MPI, left the other ranks waiting forever.
"""

# Standard Library Imports

# Third-Party Library Imports
import pytest

# Local Library Imports
from dg.mesh import Mesh
from dg.quadrature import quad_xyth

def make_mesh() -> Mesh:
    return Mesh(Ls = [1., 1.], pbcs = [False, False], ndofs = [2, 2, 2],
                has_th = True)

def test_ref_col_unsupported_kind_raises():
    mesh: Mesh = make_mesh()
    with pytest.raises(ValueError, match = "UNSUPPORTED REFINEMENT KIND"):
        mesh.ref_col(0, kind = "xyz")

def test_ref_col_unsupported_form_raises():
    mesh: Mesh = make_mesh()
    with pytest.raises(ValueError, match = "UNSUPPORTED REFINEMENT FORM"):
        mesh.ref_col(0, kind = "spt", form = "q")

def test_ref_col_spt_unsupported_form_raises():
    mesh: Mesh = make_mesh()
    with pytest.raises(ValueError, match = "UNSUPPORTED REFINEMENT FORM"):
        mesh.ref_col_spt(0, form = "q")

def test_ref_cell_unsupported_form_raises():
    mesh: Mesh = make_mesh()
    cell_key: int = sorted(mesh.cols[0].cells.keys())[0]
    with pytest.raises(ValueError, match = "UNSUPPORTED REFINEMENT FORM"):
        mesh.ref_cell(0, cell_key, form = "q")

@pytest.mark.parametrize("nnodes", [{"nnodes_x"  : 0},
                                    {"nnodes_y"  : 0},
                                    {"nnodes_th" : 0}])
def test_quad_xyth_nonpositive_nnodes_raises(nnodes):
    with pytest.raises(ValueError, match = "< 1"):
        quad_xyth(**nnodes)
