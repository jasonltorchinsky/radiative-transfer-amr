"""
Regression test for issue #13: Dirac-type boundary data (bcs_dirac = [g, [x_s, y_s, th_s]])
must reach only the elements that contain every given coordinate, at those elements'
own offsets in the column vector, and a coordinate of 0.0 is a valid location.
"""

# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports
from dg.mesh import Mesh
from rt.boundary_conditions_vector import calc_col_vec


def make_mesh() -> Mesh:
    # 2 x 2 columns, 4 cells per column, two nodes in each dimension
    mesh: Mesh = Mesh(Ls = [1., 1.], pbcs = [False, False], ndofs = [2, 2, 2],
                      has_th = True)
    mesh.ref_mesh(kind = "ang", form = "h")
    mesh.ref_mesh(kind = "ang", form = "h")
    mesh.ref_mesh(kind = "spt", form = "h")
    return mesh


def g(x, y, th) -> float:
    return 1.


def cell_blocks(mesh: Mesh, col_key: int, dirac: list) -> list:
    """ Split the column vector into (column, cell, block) triples. """
    col = mesh.cols[col_key]
    [nx, ny] = col.ndofs[:]
    col_vec: np.ndarray = calc_col_vec(mesh, col_key, [g, dirac])

    blocks: list = []
    st_idx: int = 0
    for _, cell in sorted(col.cells.items()):
        [nth] = cell.ndofs[:]
        blocks += [(col, cell, col_vec[st_idx:st_idx + nx * ny * nth])]
        st_idx += nx * ny * nth
    assert(st_idx == np.size(col_vec))
    return blocks


def all_blocks(mesh: Mesh, dirac: list) -> list:
    blocks: list = []
    for col_key in sorted(mesh.cols.keys()):
        blocks += cell_blocks(mesh, col_key, dirac)
    return blocks


def test_point_source_only_in_its_element():
    mesh: Mesh = make_mesh()
    [xs, ys, ths] = [0.25, 0.25, 7. * np.pi / 4.]

    n_src: int = 0
    for col, cell, block in all_blocks(mesh, [xs, ys, ths]):
        [x0, y0, x1, y1] = col.pos[:]
        [th0, th1] = cell.pos[:]
        if (x0 < xs < x1) and (y0 < ys < y1) and (th0 < ths < th1):
            # The source sits at the element's midpoint, where each linear
            # Lagrange basis function equals 1/2
            n_src += 1
            assert(np.allclose(block, 1. / 8.))
        else:
            assert(np.count_nonzero(block) == 0)
    assert(n_src == 1)


def test_line_source_only_in_columns_containing_it():
    mesh: Mesh = make_mesh()
    xs: float = 0.25

    for col, cell, block in all_blocks(mesh, [xs, None, None]):
        [x0, _, x1, _] = col.pos[:]
        if x0 < xs < x1:
            assert(np.allclose(block, 1. / 2.))
        else:
            assert(np.count_nonzero(block) == 0)


def test_zero_coordinate_is_a_dirac_location():
    mesh: Mesh = make_mesh()

    for col, cell, block in all_blocks(mesh, [None, None, 0.]):
        [th0, _] = cell.pos[:]
        if th0 == 0.:
            # Beam at th = 0: one angular basis function is clipped to zero,
            # so this is not the plain nodal interpolant of g
            assert(np.count_nonzero(block) == np.size(block) // 2)
        else:
            assert(np.count_nonzero(block) == 0)
