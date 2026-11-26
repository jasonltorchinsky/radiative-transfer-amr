`test_dg`
================================================================================

This subdirectory contains unit and regression tests for the code in the `src/dg` subdirectory: the *hp*-mesh hierarchy `Mesh`/`Column`/`Cell` (`dg.mesh`), projections onto the mesh (`dg.projection`), the interior-DOF masks (`dg.matrix`) and the quadrature rules (`dg.quadrature`). The `dg` package is the base layer of the method: `rt.Problem` assembles its matrices on a `dg.mesh.Mesh`, returns the solution as a `dg.projection.Projection`, and `amr.error_indicator` refines the mesh through `Mesh.ref_col` and `Mesh.ref_cell`.

The tests in this directory run serially; under `--with-mpi`, `test_get_intr_mask_mpi` is collective and the others run independently on each rank.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `test_dg` subdirectory are:

- `test_error_paths.py` : Issue #11 (PR #25), invalid refinement and quadrature requests raise `ValueError`.
- `test_helpers.py` : Issue #19 (PR #33), exported helpers that raised as soon as they were called.

`test_error_paths.py`
--------------------------------------------------------------------------------

Regression test for issue #11. Invalid arguments used to call `sys.exit(0)` or `quit()`, which reported success and, under MPI, left the other ranks waiting in a collective operation. They must now raise `ValueError`. (`test_utils/test_run_or_abort.py` checks the other half of the fix, that such an exception aborts the whole MPI job.)

Module-level helper:

- `make_mesh() -> Mesh` : one column on $[0, 1]^2$ with `ndofs = [2, 2, 2]` and one angular cell.

Tests:

- `test_ref_col_unsupported_kind_raises` : `mesh.ref_col(0, kind = "xyz")` raises `ValueError` matching `"UNSUPPORTED REFINEMENT KIND"`.
- `test_ref_col_unsupported_form_raises` : `mesh.ref_col(0, kind = "spt", form = "q")` raises `ValueError` matching `"UNSUPPORTED REFINEMENT FORM"`.
- `test_ref_col_spt_unsupported_form_raises` : the same for `mesh.ref_col_spt(0, form = "q")`.
- `test_ref_cell_unsupported_form_raises` : the same for `mesh.ref_cell(0, cell_key, form = "q")`.
- `test_quad_xyth_nonpositive_nnodes_raises[nnodes]` : `quad_xyth(nnodes_x = 0)`, `quad_xyth(nnodes_y = 0)` and `quad_xyth(nnodes_th = 0)` each raise `ValueError` matching `"< 1"`.

`test_helpers.py`
--------------------------------------------------------------------------------

Regression tests for issue #19: helpers that are exported by the `dg` packages but raised an exception as soon as they were called.

Module-level helper:

- `make_mesh() -> Mesh` : $[0, 1] \times [0, 2]$ with `ndofs = [2, 3, 2]`, refined twice in angle and once in space ($2 \times 2$ columns with 4 cells each), then the first column refined once more in angle, so that the angular meshes of neighboring columns differ in level.

Tests:

- `test_remove_th` : `dg.mesh.remove_th(mesh)` returns a mesh with `has_th = False`, the same column keys, positions, `ndofs` and neighbor keys, and a single cell (key 0, `ndofs = [1]`, `quadrant = None`) in each column.
- `test_intg_th` : for $f = (x + 2y)\theta$, which the angular Gauss rule integrates exactly, `dg.projection.intg_th.intg_th(Projection(mesh, f))` equals the projection of $\int_0^{2\pi} f \, d\theta = 2\pi^2 (x + 2y)$ onto `remove_th(mesh)` (same column keys, same `vals` shapes, values equal to round-off).
- `test_nhbr_cells_spt` : for every column, cell, axis (0, 1) and side (`"+"`, `"-"`), `mesh.nhbr_cells_spt(col_key, cell_key, axis, nhbr_loc)` returns `[[nhbr_col_1, cells_1], [nhbr_col_2, cells_2]]` where the columns are those returned by `mesh.nhbr_cols(col_key, axis, nhbr_loc)` (the same objects), `cells` is `[None, None]` for a missing neighbor, and otherwise the non-`None` entries of `cells` are exactly the leaf cells of the neighboring column whose angular interval overlaps that of the cell.
- `test_Projection_str` : `str(Projection(mesh))` contains the lines for `Ls`, `pbcs`, `has_th` and the sorted column keys.
- `test_get_intr_mask_mpi` : `dg.matrix.get_masks.get_intr_mask_mpi(mesh)` returns a boolean vector of length `mesh.get_ndof()`, not all `True`, equal to the mask of `get_intr_mask_seq(mesh)` (which is computed on the root rank only and broadcast here). Collective under MPI: both functions call `MPI.COMM_WORLD.Barrier()` and `get_intr_mask_mpi` uses `allgather`.

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `test_dg` subdirectory is

```
test_dg
|
|--- test_mesh : Unit tests for code in the src/dg/mesh subdirectory.
|    |--- test_column : Unit tests for code in the src/dg/mesh/column subdirectory.
|    |    |--- test_cell : Unit tests for code in the src/dg/mesh/column/cell subdirectory.
|--- test_projection : Unit tests for code in the src/dg/projection subdirectory.
```
