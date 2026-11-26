`projection_column`
================================================================================

This subdirectory contains the source code regarding spatial *hp*-elements in a `Projection`.

How It Fits Into the Whole
--------------------------------------------------------------------------------

A `dg.projection.Projection` stores a function on a `dg.mesh.Mesh`. For each column (spatial element $C$) of the mesh it holds one `Projection_Column` in `Projection.cols`, which copies the geometry of the corresponding `dg.mesh.column.Column` and holds the column's `Projection_Cell` objects (angular elements, subdirectory `projection_cell`). `Projection.__init__` creates the `Projection_Column` objects; `Projection.to_vector`, `Projection.from_vector`, `Projection.cell_intg_xy`, `Projection.col_intg_th`, `amr.error_indicator` and `tools.dg.projection` read them.

`__init__.py` adds the `src` directory to `sys.path` and exports the subpackage `projection_cell` and the class `Projection_Column`.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `projection_column` subdirectory are:

- `class_Projection_Column` : The class implementing spatial *hp*-elements in a `Projection`.

Functions and Classes
--------------------------------------------------------------------------------

### `class Projection_Column(col, projection_cells)` (`class_Projection_Column.py`)

- `col` (`dg.mesh.column.Column`) : The column whose attributes are copied.
- `projection_cells` (`dict`) : Cell key (`int`) to `Projection_Cell`, stored by reference in `cells`.

Attributes (copied from `col`; lists are shallow copies):

- `pos` (`list` of 4 `float`) : Corners `[x0, y0, x1, y1]` of the column.
- `idx` (`list` of 2 `int`) : Spatial index of the column at its refinement level. As in `Column.idx`, it is stored as `[y index, x index]`: `idx[0]` counts in $y$ and `idx[1]` in $x$ (issue #21; see `dg/mesh/README.md`).
- `lv` (`int`) : Spatial refinement level.
- `key` (`int`) : Unique column key; columns are ordered by increasing key in the global DOF vector.
- `is_lf` (`bool`) : Whether the column is a leaf. Always `True` in a `Projection`.
- `ndofs` (`list` `[nx, ny]`) : Number of Legendre-Gauss-Lobatto nodes in $x$ and $y$.
- `nhbr_keys` (`list` of 4 `list`s) : Neighbor column keys through the faces, in the order right ($+x$), top ($+y$), left ($-x$), bottom ($-y$). Each entry is `[key_0, key_1]`: two different keys when the neighbor on that face is refined once more, the same key twice when there is a single neighbor (of the same or a coarser level), and `[None, None]` on a non-periodic domain boundary. The outer list is copied, the inner lists are shared with the mesh in `Projection.mesh`.
- `cells` (`dict`) : Cell key to `Projection_Cell`.

Methods:

- `__eq__(other)` : `True` if `pos`, `idx`, `lv`, `key`, `is_lf`, `ndofs`, `nhbr_keys` and `cells` are all equal (the cells are compared with `Projection_Cell.__eq__`).
- `__str__()` : Prints `idx`, `lv`, `key`, `pos`, `ndofs` and the cell keys.

Example:

```python
proj_col = proj.cols[col_key]
[x0, y0, x1, y1] = proj_col.pos
for cell_key, proj_cell in sorted(proj_col.cells.items()):
    print(cell_key, proj_cell.vals.shape) # (nx, ny, nth)
```

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `projection_column` subdirectory is

```
projection_column
|
|--- projection_cell : Contains source code implementing the Projection_Cell class.
```
