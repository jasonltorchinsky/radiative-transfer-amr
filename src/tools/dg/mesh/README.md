`mesh`
================================================================================

This subdirectory contains tools for visualizing an *hp*-mesh (`dg.mesh.Mesh`): the spatial-angular *hp*-mesh with the number of degrees of freedom of each element, and the neighbors of a chosen spatial or spatial-angular element.

How It Fits Into the Whole
--------------------------------------------------------------------------------

`plot_mesh` is called by `experiments/*/generate_experiment_visualizations.py` for the mesh of each refinement step (`mesh.png`) and of the high-resolution solution (`mesh_hr.png`), and by `experiments/test-rt/experiment.py`. Figures of this kind are Fig. 9 and the final-mesh figures of the Supplementary Information of the publication. `plot_nhbrs` is a debugging aid for the neighbor relations used in assembly and refinement (the visualization of Fig. 1(c)); no experiment script calls it.

Both functions read `Mesh.Ls`, `Mesh.cols`, `Mesh.get_ndof()`, `Column.pos`, `Column.ndofs`, `Column.cells`, `Cell.pos` and `Cell.ndofs`; `plot_nhbrs` also uses `Column.nhbr_keys`, `Cell.nhbr_keys` and `Mesh.nhbr_cells_in_nhbr_col` (see `dg/mesh/README.md` for these conventions). They import `consts` (for `PI`) and `dg.mesh`. They do not modify the mesh and do not use MPI.

Package Initialization
--------------------------------------------------------------------------------

`__init__.py` defines `src_dir` (`str`) and appends it to `sys.path` if absent (the computed path is `src/src`, which does not exist, so `src` must already be on `sys.path`), and exports:

- `plot_mesh` (`plot_mesh.py`)
- `plot_nhbrs` (`plot_nhbrs.py`)

Source Files
--------------------------------------------------------------------------------

The source files contained in the `mesh` subdirectory are:

- `plot_mesh` : Visualizes an *hp*-mesh, including the number of degrees of freedom in the $x$- (same as the $y$-), and in the $\theta$-dimension.
- `plot_nhbrs` : Visualizes the neighbors of a given spatial or spatial-angular *hp*-element.

Both draw each leaf column as a rectangle `[x0, x1] x [y0, y1]` and its angular mesh as a disk of radius $\min(\Delta x, \Delta y)/2$ centred in the rectangle, with one wedge per leaf cell from $\theta_0$ to $\theta_1$ (degrees, counterclockwise from $+x$). All patches have black edges. The axes are labelled `x` and `y` with limits $[0, L_x] \times [0, L_y]$ unless `lims` is given, and the title is `"Number of Degress of Freedom: N"` (spelling as in the code), where $N$ = `mesh.get_ndof()` $= \sum_{(C, K)} n_x n_y n_\theta$ over all leaf columns and leaf cells.

### `plot_mesh(mesh, file_path = None, **kwargs)` (`plot_mesh.py`)

Plots the spatial-angular *hp*-mesh, coloured by the number of nodes per element.

- `mesh` (`dg.mesh.Mesh`) : the mesh; every column and cell must be a leaf.
- `file_path` (`str` or `None`, default `None`) : if given, the figure is saved there (6.5 in $\times$ $6.5 L_y / L_x$ in, 300 dpi) and closed.
- `lims` (`list`, default `[[], []]`) : `[[x_min, x_max], [y_min, y_max]]`; an empty list means the full range.
- `show_p` (`bool`, default `True`) : if `True`, each rectangle is filled with the colour for its number of $x$-nodes `ndofs[0]` (the number of $y$-nodes is not shown; the publication's Fig. 9 caption notes it equals the $x$ value), each wedge with the colour for its number of $\theta$-nodes `ndofs[0]` of the cell, and a legend lists each distinct node count (spatial and angular counts share one palette and one legend, sorted ascending). If `False`, all patches are unfilled and there is no legend.
- `blocking` (`bool`, default `False`) : accepted but not used.

The colour for a node count $n$ is entry $n \bmod 17$ of a fixed list of 17 colours (`"#e6194B"`, `"#f58231"`, `"#ffe119"`, ..., `"#800000"`), so counts that differ by a multiple of 17 share a colour. Wedges are drawn on top of the rectangles.

Returns `[fig, ax]` (`matplotlib.figure.Figure`, `matplotlib.axes.Axes`).

```python
from dg.mesh import Mesh
from tools.dg.mesh import plot_mesh

mesh = Mesh(Ls = [2., 1.], pbcs = [False, False], ndofs = [3, 3, 4], has_th = True)
mesh.ref_mesh(kind = "all", form = "h")
plot_mesh(mesh, "mesh.png")
```

### `plot_nhbrs(mesh, col_key, cell_key, file_path = None, **kwargs)` (`plot_nhbrs.py`)

Highlights a column or a spatial-angular element and its neighbors.

- `mesh` (`dg.mesh.Mesh`) : the mesh; every column and cell must be a leaf.
- `col_key` (`int`) : key of the column.
- `cell_key` (`int` or `None`) : key of a cell in that column, or `None` to show the neighbors of the column.
- `file_path` (`str` or `None`, default `None`) : if given, the figure is saved there (6.5 in $\times$ $6.5 L_y / L_x$ in, 300 dpi) and closed.
- `lims` (`list`, default `[[], []]`) : as in `plot_mesh`.
- `blocking` (`bool`, default `False`) : accepted but not used.

The spatial neighbors are all distinct non-`None` keys in `mesh.cols[col_key].nhbr_keys` (all four faces, including periodic neighbors).

- `cell_key is None` : the column's rectangle is filled orange (`"#E69F00"`) and the rectangles of its spatial neighbors green (`"#009E73"`); all wedges are unfilled.
- `cell_key` given : all rectangles are unfilled. The cell's wedge is orange (`"#E69F00"`); the wedges of its angular neighbors in the same column (`Cell.nhbr_keys`) are light blue (`"#56B4E9"`); and in each spatially neighboring column, the cells returned by `mesh.nhbr_cells_in_nhbr_col(col_key, cell_key, nhbr_col_key)` (the leaf cells that overlap the cell in angle: same level, parent, or the two children) are green (`"#009E73"`).

Fig. 1(c) of the publication uses a different colour scheme for the same picture (element in black, neighbors within the angular mesh in orange, neighbors in spatially adjacent angular meshes in blue).

Returns `[fig, ax]`.

```python
from tools.dg.mesh import plot_nhbrs

col_key = sorted(mesh.cols.keys())[0]
cell_key = sorted(mesh.cols[col_key].cells.keys())[0]
plot_nhbrs(mesh, col_key, cell_key, "nhbrs_cell.png") # one spatial-angular element
plot_nhbrs(mesh, col_key, None, "nhbrs_col.png")      # one column
```
