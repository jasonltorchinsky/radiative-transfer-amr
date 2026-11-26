"""
Regression test for issue #15: p-refinement must enforce the even-degree
criterion of Sec. 3.2 of the paper. If a spatial-angular hp-element (C, K) has
multiple spatial-angular hp-elements neighboring through a face of (C, K), then
each of those neighboring elements must have the same number of spatial and of
angular degrees of freedom.
"""

# Standard Library Imports
import random

# Third-Party Library Imports

# Local Library Imports
from dg.mesh import Mesh

def assert_even_degree(mesh: Mesh) -> None:
    for col_key, col in mesh.cols.items():
        for col_face in range(0, 4):
            nhbr_col_keys: list = [nhbr_col_key
                                   for nhbr_col_key in set(col.nhbr_keys[col_face])
                                   if nhbr_col_key is not None]

            ## Spatial degrees of freedom of the columns behind the face
            spt_ndofs: set = {tuple(mesh.cols[nhbr_col_key].ndofs)
                              for nhbr_col_key in nhbr_col_keys}
            assert len(spt_ndofs) <= 1, \
                "column {} face {}: neighbours {} have ndofs {}".format(
                    col_key, col_face, nhbr_col_keys, spt_ndofs)

            ## Angular degrees of freedom of the cells behind the face
            for cell_key in col.cells.keys():
                ang_ndofs: dict = {}
                for nhbr_col_key in nhbr_col_keys:
                    nhbr_cell_keys: list = \
                        mesh.nhbr_cells_in_nhbr_col(col_key, cell_key,
                                                    nhbr_col_key)
                    for nhbr_cell_key in set(nhbr_cell_keys):
                        if nhbr_cell_key is not None:
                            nhbr_cell = mesh.cols[nhbr_col_key].cells[nhbr_cell_key]
                            ang_ndofs[(nhbr_col_key, nhbr_cell_key)] = \
                                nhbr_cell.ndofs[0]
                assert len(set(ang_ndofs.values())) <= 1, \
                    "column {} cell {} face {}: neighbours have n_theta {}".format(
                        col_key, cell_key, col_face, ang_ndofs)

def test_even_degree_spt_p():
    # 2 x 2 columns, refine the bottom-left column in h, then p-refine one of
    # its two children behind the left face of the bottom-right column.
    mesh: Mesh = Mesh(Ls = [2., 2.], pbcs = [False, False], ndofs = [3, 3, 3],
                      has_th = True)
    mesh.ref_mesh(kind = "ang", form = "h")
    mesh.ref_mesh(kind = "ang", form = "h")
    mesh.ref_mesh(kind = "spt", form = "h")
    mesh.ref_col(1, kind = "spt", form = "h")

    right_col_key: int = 3
    left_face_keys: list = mesh.cols[right_col_key].nhbr_keys[2][:]
    assert len(set(left_face_keys)) == 2

    mesh.ref_col(left_face_keys[0], kind = "spt", form = "p")

    for col_key in left_face_keys:
        assert mesh.cols[col_key].ndofs == [4, 4]
    assert_even_degree(mesh)

def test_even_degree_ang_p():
    # 1 x 2 columns, refine one angular cell of the left column in h, then
    # p-refine one of its two children.
    mesh: Mesh = Mesh(Ls = [2., 1.], pbcs = [False, False], ndofs = [3, 3, 3],
                      has_th = True)
    mesh.ref_mesh(kind = "ang", form = "h")
    mesh.ref_mesh(kind = "ang", form = "h")
    mesh.ref_col(0, kind = "spt", form = "h")

    [left_col_key, right_col_key] = [1, 3]
    mesh.ref_cell(left_col_key, 4, form = "h")
    mesh.ref_cell(left_col_key, 9, form = "p")

    nhbr_cell_keys: list = mesh.nhbr_cells_in_nhbr_col(right_col_key, 4,
                                                       left_col_key)
    assert sorted(nhbr_cell_keys) == [9, 10]
    for cell_key in nhbr_cell_keys:
        assert mesh.cols[left_col_key].cells[cell_key].ndofs == [4]
    assert_even_degree(mesh)

def test_even_degree_random_refinements():
    # The criterion holds after every step of a sequence of h-, p-, and
    # hp-refinements in space and in angle.
    rng: random.Random = random.Random(15)
    for pbcs in [[False, False], [True, True]]:
        mesh: Mesh = Mesh(Ls = [1., 1.], pbcs = pbcs, ndofs = [2, 2, 2],
                          has_th = True)
        mesh.ref_mesh(kind = "ang", form = "h")
        mesh.ref_mesh(kind = "spt", form = "h")
        assert_even_degree(mesh)

        for _ in range(0, 40):
            col_key: int = rng.choice(sorted(mesh.cols.keys()))
            form: str = rng.choice(["h", "p", "hp"])
            if rng.random() < 0.5:
                if (form != "p") and (mesh.cols[col_key].lv >= 3):
                    form = "p"
                mesh.ref_col(col_key, kind = "spt", form = form)
            else:
                col = mesh.cols[col_key]
                cell_key: int = rng.choice(sorted(col.cells.keys()))
                if (form != "p") and (col.cells[cell_key].lv >= 4):
                    form = "p"
                mesh.ref_cell(col_key, cell_key, form = form)
            assert_even_degree(mesh)
