`test_projection`
================================================================================

This subdirectory contains the unit tests for numerical projections on an *hp*-mesh, the `Projection` class of `src/dg/projection`. A `Projection` holds the nodal values of a function on every element $(C, K)$ of a `Mesh` (`proj.cols[col_key].cells[cell_key].vals`, an array of shape `[nx, ny, nth]` indexed `[i, j, a]` over the LGL nodes in $x$ and $y$ and the LG nodes in $\theta$). `rt.Problem.solve` returns its solution as a `Projection`, `amr.error_indicator.Error_Indicator` computes its indicators from one, and the plotting functions of `src/tools/dg/projection` draw one.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `test_projection` subdirectory are:

- `test_Projection.py` : Unit test for the `Projection` class (construction, file round trip, plotting).

`test_Projection.py`
--------------------------------------------------------------------------------

- `test_Projection(tmp_path)` : builds `Mesh(Ls = [1., 2.], pbcs = [False, False], ndofs = [3, 3, 3], has_th = True)`, applies `mesh.ref_mesh(kind = "all", form = "hp")` four times, plots it with `tools.dg.mesh.plot_mesh` to `mesh.png`, and projects $f(x, y, \theta) = \cos\theta \, (x + y)$ onto it with `Projection(mesh, f)`. It then
  1. writes the projection with `proj.to_file("proj.npy", write_mesh = True, mesh_file_path = "mesh.json")`, reads it back with `dg.projection.from_file(mesh_file_path, proj_file_path)`, and asserts that the two projections are equal (`Projection.__eq__` compares the meshes and, cell by cell, the attributes and `vals` exactly);
  2. plots the projection with `plot_th`, `plot_xth`, `plot_yth`, `plot_xy` and `plot_xyth` from `tools.dg.projection` to `proj_th.png`, `proj_xth.png`, `proj_yth.png`, `proj_xy.png` and `proj_xyth.png`.

  All files go to `tmp_path`. `Projection.to_file` writes the values with `numpy.ndarray.tofile`, i.e., as raw `float64` without a `.npy` header despite the file name, and `from_file` reads them with `numpy.fromfile`. The test checks that projection, I/O and plotting run and that the round trip is exact; it does not check the projected values against $f$.
