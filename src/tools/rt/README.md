`rt`
================================================================================

This subdirectory contains tools for visualizing matrices: it plots the sparsity pattern of a matrix, such as the mass (extinction), scattering, convection and system matrices assembled by the solver package `rt`.

How It Fits Into the Whole
--------------------------------------------------------------------------------

`rt.Problem` assembles the DG system matrices as distributed `petsc4py` matrices (Eqs. 10-11 of the publication). `experiments/test-rt/experiment.py` obtains the mass, scattering, interior convection, boundary convection and system matrices from its `Problem`, extracts dense blocks of each on the root process (`if comm_rank == consts.COMM_ROOT`): the rows and columns owned by that process (`mat[r0:r1, r0:r1]` with `(r0, r1) = mat.getOwnershipRange()`), the leading block of the column with the smallest key, and the leading block of that column's cell with the smallest key, and passes each to `plot_matrix` (`figs/<matrix>_global.png`, `figs/<matrix>_column.png`, `figs/<matrix>_cell.png` with `<matrix>` one of `mass`, `scattering`, `interior_convection`, `boundary_convection`, `system`). Because `dg.matrix` orders the DOFs by column key, then by cell key within a column, with $\theta$ fastest within an element, the leading block of a matrix belongs to the first column, and the leading block of that to its first cell.

`plot_matrix.py` imports `numpy`, `matplotlib` and `dg.mesh.Mesh` (the import is not used). It does not use MPI.

Package Initialization
--------------------------------------------------------------------------------

`__init__.py` defines `src_dir` (`str`) and appends it to `sys.path` if absent (the computed path is `src/src`, which does not exist, so `src` must already be on `sys.path`), and exports `plot_matrix` (`plot_matrix.py`). `import tools` does not import `tools.rt`; import it explicitly (`from tools.rt import plot_matrix`).

Source Files
--------------------------------------------------------------------------------

The source files contained in the `rt` subdirectory are:

- `plot_matrix` : Visualizes the sparsity pattern of a matrix.

### `plot_matrix(matrix, file_path = "out.png", **kwargs)` (`plot_matrix.py`)

Plots the nonzero entries of `matrix` with `matplotlib.axes.Axes.spy` as black square markers (row index increasing downward, column index to the right, as `spy` draws them) and saves the figure.

- `matrix` (2-D `numpy.ndarray`, or any matrix `Axes.spy` accepts, such as a `scipy.sparse` matrix) : the matrix. A dense array has a marker at every entry that is not exactly zero.
- `file_path` (`str`, default `"out.png"`) : output image path. The figure is always saved (6.5 in $\times$ 6.5 in, 300 dpi, `bbox_inches = "tight"`) and then closed.
- `title` (`str` or `None`, default `None`) : axes title, if given.
- `marker_size` (`float`, default `2.`) : marker size in points.

Returns `None`. Unlike the other plotting functions, it does not return the figure, and it writes `out.png` in the current directory when `file_path` is not given.

```python
from tools.rt import plot_matrix

# mat: a petsc4py.PETSc.Mat assembled by rt.Problem
(r0, r1) = mat.getOwnershipRange()
plot_matrix(mat[r0:r1, r0:r1], "mass_global.png",
            title = "Global Mass Matrix", marker_size = 0.5)
```
