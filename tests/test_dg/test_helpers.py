"""
Regression tests for issue #19: helpers exported by the dg packages that raised
as soon as they were called (remove_th/intg_th, Mesh.nhbr_cells_spt,
Projection.__str__, get_intr_mask_mpi).
"""

# Standard Library Imports

# Third-Party Library Imports
import numpy as np
from mpi4py import MPI

# Local Library Imports
import consts
from dg.mesh import Mesh, remove_th
from dg.mesh.column.cell import Cell
from dg.projection import Projection
from dg.projection.intg_th import intg_th
from dg.matrix.get_masks import get_intr_mask_seq, get_intr_mask_mpi

# Relative Imports

def make_mesh() -> Mesh:
    # Spatially refined once, angularly refined twice, and one column refined
    # angularly once more so that neighboring cells differ in level
    Ls: list = [1., 2.]
    pbcs: list = [False, False]
    ndofs: list = [2, 3, 2]
    has_th: bool = True

    mesh: Mesh = Mesh(Ls, pbcs, ndofs, has_th)
    mesh.ref_mesh(kind = "ang", form = "h")
    mesh.ref_mesh(kind = "ang", form = "h")
    mesh.ref_mesh(kind = "spt", form = "h")
    col_key: int = sorted(mesh.cols.keys())[0]
    mesh.ref_col(col_key, kind = "ang", form = "h")

    return mesh

def test_remove_th():
    mesh: Mesh = make_mesh()
    mesh_flat: Mesh = remove_th(mesh)

    assert(not mesh_flat.has_th)
    assert(sorted(mesh_flat.cols.keys()) == sorted(mesh.cols.keys()))
    for col_key, col in sorted(mesh.cols.items()):
        col_flat = mesh_flat.cols[col_key]
        assert(col_flat.pos == col.pos)
        assert(col_flat.ndofs == col.ndofs)
        assert(col_flat.nhbr_keys == col.nhbr_keys)
        assert(list(col_flat.cells.keys()) == [0])
        cell: Cell = col_flat.cells[0]
        assert(cell.ndofs == [1])
        assert(cell.quadrant is None)

def test_intg_th():
    mesh: Mesh = make_mesh()

    # Linear in th, so the angular quadrature is exact
    def f(x, y, th):
        return (x + 2. * y) * th
    proj: Projection = Projection(mesh, f)
    proj_intg_th: Projection = intg_th(proj)

    def f_intg_th(x, y):
        return (x + 2. * y) * 2. * consts.PI**2
    proj_ref: Projection = Projection(remove_th(mesh), f_intg_th)

    assert(not proj_intg_th.mesh.has_th)
    assert(sorted(proj_intg_th.cols.keys()) == sorted(proj_ref.cols.keys()))
    for col_key, proj_col in sorted(proj_ref.cols.items()):
        vals: np.ndarray = proj_intg_th.cols[col_key].cells[0].vals
        vals_ref: np.ndarray = proj_col.cells[0].vals
        assert(vals.shape == vals_ref.shape)
        assert(np.allclose(vals, vals_ref))

def test_nhbr_cells_spt():
    mesh: Mesh = make_mesh()

    for col_key, col in sorted(mesh.cols.items()):
        for cell_key, cell in sorted(col.cells.items()):
            [th0, th1] = cell.pos
            for axis in [0, 1]:
                for nhbr_loc in ["+", "-"]:
                    [_, nhbr_col_1, nhbr_col_2] = \
                        mesh.nhbr_cols(col_key, axis, nhbr_loc)
                    nhbr_cells: list = mesh.nhbr_cells_spt(col_key, cell_key,
                                                           axis, nhbr_loc)

                    assert(nhbr_cells[0][0] is nhbr_col_1)
                    assert(nhbr_cells[1][0] is nhbr_col_2)
                    for [nhbr_col, cells] in nhbr_cells:
                        if nhbr_col is None:
                            assert(cells == [None, None])
                            continue
                        # Leaf cells of the neighboring column whose angular
                        # interval overlaps that of the cell
                        overlap: list = [nhbr_cell.key for nhbr_cell
                                         in nhbr_col.cells.values()
                                         if (nhbr_cell.is_lf
                                             and nhbr_cell.pos[0] < th1
                                             and nhbr_cell.pos[1] > th0)]
                        found: list = [nhbr_cell.key for nhbr_cell in cells
                                       if nhbr_cell is not None]
                        assert(sorted(found) == sorted(overlap))

def test_Projection_str():
    mesh: Mesh = make_mesh()
    proj: Projection = Projection(mesh)

    msg: str = str(proj)
    assert("Ls     :  {}\n".format(mesh.Ls) in msg)
    assert("pbcs   :  {}\n".format(mesh.pbcs) in msg)
    assert("has_th :  {}\n".format(mesh.has_th) in msg)
    assert("cols   :  {}\n".format(sorted(mesh.cols.keys())) in msg)

def test_get_intr_mask_mpi():
    mpi_comm: MPI.Intracomm = MPI.COMM_WORLD
    mesh: Mesh = make_mesh()

    # Sequential version returns the mask on the root rank only
    mask_seq: np.ndarray = get_intr_mask_seq(mesh)
    mask_seq: np.ndarray = mpi_comm.bcast(mask_seq, root = consts.COMM_ROOT)

    mask_mpi: np.ndarray = get_intr_mask_mpi(mesh)

    assert(mask_mpi.dtype == bool)
    assert(mask_mpi.shape == (mesh.get_ndof(),))
    assert(not np.all(mask_mpi))
    assert(np.array_equal(mask_mpi, mask_seq))
