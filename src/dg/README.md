`dg`
================================================================================

This subdirectory contains the source code regarding the discontinuous Galerkin spectral element method (DGSEM): the spatial-angular *hp*-mesh, functions represented on that mesh, numerical quadrature and Lagrange polynomials, and the handling of global matrices and vectors spread across MPI processes. It contains no radiative-transfer-specific code; the radiative transfer equation is assembled from these pieces in `rt`.

How It Fits Into the Whole
--------------------------------------------------------------------------------

`dg` sits between the constants and utilities (`consts`, `utils`) and the solver and adaptivity layers (`rt`, `amr`, `tools`):

- `rt.Problem` takes a `dg.mesh.Mesh`, loops over its columns and cells using the quadrature rules of `dg.quadrature` and the maps of `dg.projection` (`push_forward`, `pull_back`; `get_f2f_matrix` is imported by `rt/get_Ey.py` but not called), indexes degrees of freedom with `dg.matrix` (`get_idx_map`, `get_col_idxs`, `get_cell_idxs`), separates the inflow-boundary DOFs with `dg.matrix` (`get_intr_mask`, `split_matrix`, `split_vector`, `merge_vectors`), and returns the solution as a `dg.projection.Projection`.
- `amr.error_indicator.Error_Indicator` reads a `Projection` (and its `Mesh`), evaluates it with `dg.quadrature`, and returns a refined `Mesh` by calling `Mesh.ref_col` and `Mesh.ref_cell`.
- `tools` plots `Mesh` and `Projection` objects.

Data Model
--------------------------------------------------------------------------------

The mesh is '2 + 1'-dimensional (Section 2.1 of the publication):

- `mesh.Mesh` : the spatial domain $[0, L_x] \times [0, L_y]$ (`Mesh.Ls`), with columns in the dictionary `Mesh.cols` keyed by column key.
- `mesh.column.Column` : a spatial element $C$ (rectangle `pos = [x0, y0, x1, y1]`), with spatial degrees of freedom `ndofs = [nx, ny]`, neighbor keys through each of its four faces, and its angular mesh in the dictionary `Column.cells`.
- `mesh.column.cell.Cell` : an angular element $K$ (arc `pos = [th0, th1]` in $[0, 2\pi]$) with angular degrees of freedom `ndofs = [nth]`.

A function on the mesh is a `projection.Projection`, which mirrors this hierarchy (`Projection_Column`, `Projection_Cell`). Each `Projection_Cell` stores the nodal values $u^{C,K}_{ija}$ of Eq. 6 in `vals`, an array of shape `[nx, ny, nth]` indexed `[i, j, a]`, at the LGL nodes in $x$ and $y$ and the LG nodes in $\theta$ returned by `quadrature.quad_xyth` and mapped to the element with `projection.push_forward`. Flattened per element, index $\alpha = n_\theta n_y i + n_\theta j + a$ ($\theta$ fastest, `matrix.get_idx_map`); globally, leaf columns follow in increasing key order and, within a column, leaf cells in increasing key order.

Package Initialization
--------------------------------------------------------------------------------

`__init__.py` defines `src_dir` (`str`, the normalized path of the parent `src` directory), appends it to `sys.path` if absent, and imports the four subpackages, so after `import dg` the names below are available:

- `dg.matrix` : subpackage; see `matrix/README.md`.
- `dg.mesh` : subpackage; see `mesh/README.md`.
- `dg.projection` : subpackage; see `projection/README.md`.
- `dg.quadrature` : subpackage; see `quadrature/README.md`.

`dg/__init__.py` defines no functions or classes of its own. The main public names of the subpackages are:

| Subpackage | Exported names |
| --- | --- |
| `dg.mesh` | `Mesh` (methods include `ref_mesh`, `ref_col`, `ref_cell`, `get_ndof`, `nhbr_cols`, `nhbr_cells_spt`, `to_file`), `from_file`, `remove_th`, subpackage `column` (`Column`, `calc_key`, subpackage `cell` with `Cell`, `calc_key`) |
| `dg.projection` | `Projection` (methods include `to_vector`, `from_vector`, `cell_intg_xy`, `col_intg_th`, `to_file`), `from_file`, `get_f2f_matrix`, `push_forward`, `pull_back`, subpackage `projection_column` |
| `dg.matrix` | `get_idx_map`, `get_idx_inv_map`, `get_col_idxs`, `get_cell_idxs`, `get_intr_mask`, `get_bdry_mask`, `split_matrix`, `split_vector`, `merge_vectors` |
| `dg.quadrature` | `quad_xyth`, `lg_quad`, `lgl_quad`, `lgr_quad`, `uni_quad`, `lag_eval`, `lag_ddx`, `lag_ddx_eval`, `calc_proj_mtx_1d`, `proj_1d` |

Example:

```python
import numpy as np
from dg.mesh import Mesh
from dg.projection import Projection

mesh = Mesh(Ls = [2., 1.], pbcs = [False, False], ndofs = [3, 3, 4], has_th = True)
mesh.ref_mesh(kind = "ang", form = "h") # Split every angular cell in two
proj = Projection(mesh, lambda x, y, th: np.exp(-x) * np.cos(th)**2)
vec = proj.to_vector()                  # Length mesh.get_ndof()
```

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `dg` subdirectory is

```
dg
|
|--- matrix : Contains source code to index degrees of freedom and to handle matrices and vectors spread across multiple MPI processes (interior/boundary masks, splitting and merging).
|--- mesh : Contains source code to handle the *hp*-mesh (the Mesh, Column, and Cell classes, neighbor queries, h- and p-refinement, and file I/O).
|--- projection : Contains source code to handle functions projected to the *hp*-mesh (the Projection class, push-forward and pull-back maps, face-to-face matrices, integration, and file I/O).
|--- quadrature : Contains source code to handle numerical quadrature (Legendre-Gauss, Legendre-Gauss-Lobatto, Legendre-Gauss-Radau, and uniform rules) and Lagrange polynomials.
```
