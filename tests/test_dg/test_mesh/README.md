`test_mesh`
================================================================================

This subdirectory contains the unit tests for the *hp*-mesh, the `Mesh` class of `src/dg/mesh`. A `Mesh` holds a dictionary of spatial elements (`Column`s, keyed by `col_key`), each with its own dictionary of angular elements (`Cell`s, keyed by `cell_key`). Every other part of the code works on a `Mesh`: `rt.Problem.solve` assembles and solves on it, `dg.projection.Projection` stores values on it, and `amr.error_indicator.Error_Indicator.ref_by_ind` refines it with `Mesh.ref_col` and `Mesh.ref_cell`, which also enforce 1-irregularity and the even-degree criterion.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `test_mesh` subdirectory are:

- `test_Mesh.py` : Unit test for the `Mesh` class (string output, file round trip, refinement, plotting).
- `test_even_degree.py` : Issue #15 (PR #29), *p*-refinement enforces the even-degree criterion.

`test_Mesh.py`
--------------------------------------------------------------------------------

- `test_Mesh(tmp_path)` : builds `Mesh(Ls = [1., 2.], pbcs = [False, False], ndofs = [2, 3, 4], has_th = True)` (one column, one cell) and
  1. writes `mesh.__str__()` to `mesh_str.txt`;
  2. writes the mesh with `mesh.to_file("mesh.json")`, reads it back with `dg.mesh.from_file`, and asserts that the two meshes are equal (`Mesh.__eq__`);
  3. plots the mesh with `tools.dg.mesh.plot_mesh` to `mesh_0.png`;
  4. refines columns `[0, 0, 0, 1, 10, 48]` with `(kind, form)` = `("ang", "h")`, `("ang", "h")`, `("spt", "h")`, `("spt", "h")`, `("spt", "hp")`, `("spt", "hp")` through `Mesh.ref_col`, then cells `[3, 7, 15]` of column 204 with forms `"h"`, `"hp"`, `"hp"` through `Mesh.ref_cell`, plotting the mesh with `show_p = True` after each step (`mesh_1.png` ... `mesh_9.png`);
  5. asserts that every column and cell left in the mesh is a leaf (`is_lf`), and plots the neighbors of every column (`plot_nhbrs(mesh, col_key, None)`, `nhbrs_<col>_NONE.png`) and of every cell (`nhbrs_<col>_<cell>.png`).

  The hard-coded keys follow the key formulas of `dg.mesh.column.calc_key` (column at level $\ell$ with `idx` $= [i_0, i_1]$: $\sum_{l < \ell} 4^l + 2^\ell i_1 + i_0$; note that `Mesh.ref_col_spt` increments $i_0$ with $y$ and $i_1$ with $x$, so a column's key is $\sum_{l < \ell} 4^l + 2^\ell \times (x\text{-index}) + (y\text{-index})$) and `dg.mesh.column.cell.calc_key` (cell at level $\ell$ with index $i$: $2^\ell - 1 + i$). Column 10 is a child of column 1, column 48 a child of column 10, and column 204 a child of column 48; cells 3, 7 and 15 are the first cells at angular levels 2, 3 and 4. The test checks that refinement, I/O and plotting run without error; apart from the round trip and the leaf check it does not check the resulting mesh. Because it writes one plot per column and per cell, it is the slowest test of the suite.

`test_even_degree.py`
--------------------------------------------------------------------------------

Regression test for issue #15. Sec. 3.2 of the paper requires that if a spatial-angular element $(C, K)$ has several neighboring elements across one face, those neighbors all have the same number of spatial and of angular degrees of freedom. Before the fix, *p*-refinement of one of two such neighbors did not raise the other.

Module-level helper:

- `assert_even_degree(mesh)` : for every column and each of its four faces (index 0-3, right face first and counterclockwise, as in `Column.nhbr_keys`), asserts that all neighboring columns behind the face have the same `ndofs`, and, for every cell of the column, that all neighboring cells found with `mesh.nhbr_cells_in_nhbr_col(col_key, cell_key, nhbr_col_key)` in those columns have the same $n_\theta$.

Tests:

- `test_even_degree_spt_p` : $2 \times 2$ columns on $[0, 2]^2$ with `ndofs = [3, 3, 3]` (keys 1, 2, 3, 4 at $[0,1]^2$, $[0,1] \times [1,2]$, $[1,2] \times [0,1]$, $[1,2]^2$); the bottom-left column 1 is *h*-refined so that the left face of the bottom-right column 3 has two neighbors (columns 9 and 10). *p*-refining one of them must raise both to `ndofs = [4, 4]`, and `assert_even_degree` must hold.
- `test_even_degree_ang_p` : $2 \times 2$ columns on $[0, 2] \times [0, 1]$ with four cells each (one per quadrant); cell 4 ($[\pi/2, \pi]$) of the bottom-left column (key 1) is *h*-refined into cells 9 and 10, which both neighbor cell 4 of the bottom-right column (key 3). *p*-refining cell 9 must raise both 9 and 10 to $n_\theta = 4$, and `assert_even_degree` must hold.
- `test_even_degree_random_refinements` : for non-periodic and periodic boundaries, starts from $2 \times 2$ columns with two cells each and applies 40 random `ref_col` (spatial) or `ref_cell` refinements of form `"h"`, `"p"` or `"hp"` (seeded `random.Random(15)`; forced to `"p"` beyond spatial level 3 or angular level 4 to keep the mesh small), checking `assert_even_degree` after every step.

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `test_mesh` subdirectory is

```
test_mesh
|
|--- test_column : Contains unit tests for the Column class.
|    |--- test_cell : Contains unit tests for the Cell class.
```
