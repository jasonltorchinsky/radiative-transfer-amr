`projection_cell`
================================================================================

This subdirectory contains the source code regarding angular *hp*-elements in a `Projection`.

How It Fits Into the Whole
--------------------------------------------------------------------------------

Each `dg.projection.projection_column.Projection_Column` holds, in its `cells` dictionary, one `Projection_Cell` per angular element $K$ of the column. A `Projection_Cell` copies the attributes of the corresponding `dg.mesh.column.cell.Cell` and stores the nodal values of the function on the spatial-angular *hp*-element $(C, K)$ in `vals`. `Projection.__init__` and `Projection.from_vector` write `vals`; `Projection.to_vector`, the integrals `cell_intg_xy` and `col_intg_th`, `amr.error_indicator` and `tools.dg.projection` read it.

`__init__.py` adds the `src` directory to `sys.path` and exports the class `Projection_Cell`.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `projection_cell` subdirectory are:

- `class_Projection_Cell` : The class implementing angular *hp*-elements in a `Projection`.

Functions and Classes
--------------------------------------------------------------------------------

### `class Projection_Cell(cell, vals)` (`class_Projection_Cell.py`)

- `cell` (`dg.mesh.column.cell.Cell`) : The cell whose attributes are copied.
- `vals` (`numpy` `ndarray`, shape `[nx, ny, nth]`) : Nodal values, stored by reference.

Attributes:

- `pos` (`list` `[th0, th1]`) : Angular interval of the cell, a sub-interval of $[0, 2\pi]$ (copied). `[0, 0]` for a mesh without the angular dimension.
- `idx` (`int`) : Angular index of the cell at its refinement level.
- `lv` (`int`) : Angular refinement level.
- `key` (`int`) : Unique cell key within the column; cells are ordered by increasing key within a column in the global DOF vector.
- `is_lf` (`bool`) : Whether the cell is a leaf. Always `True` in a `Projection`.
- `ndofs` (`list` `[nth]`) : Number of Legendre-Gauss nodes in $\theta$ (copied).
- `quadrant` (`int` or `None`) : Quadrant $q \in \{0, 1, 2, 3\}$ with $[\theta_0, \theta_1] \subseteq [q \pi / 2, (q + 1) \pi / 2]$, or `None` if the cell spans more than one quadrant.
- `nhbr_keys` (`list` of 2) : Keys of the angular neighbors in the same column, `[lower-angle neighbor, upper-angle neighbor]` (the angular domain is periodic). This is the same list object as the `nhbr_keys` of the `Cell` in `Projection.mesh`, not a copy.
- `vals` (`ndarray`, shape `[nx, ny, nth]`) : `vals[i, j, a]` is the value at the node $(x_i, y_j, \theta_a)$, with Legendre-Gauss-Lobatto nodes in $x$ and $y$ and Legendre-Gauss nodes in $\theta$, all in increasing order (see `dg/projection/README.md`, Data Layout). These are the coefficients $u^{C,K}_{ija}$ of Eq. 6 of the publication.

Methods:

- `__eq__(other)` : `True` if `pos`, `idx`, `lv`, `key`, `is_lf`, `ndofs`, `quadrant`, `nhbr_keys` are equal and `vals` are exactly equal elementwise. It does not check whether the two cells belong to the same column. All comparisons are evaluated before they are combined, so cells whose `vals` have different shapes raise a `ValueError` from `numpy` instead of returning `False`.
- `__str__()` : Prints `idx`, `lv`, `key`, `pos` (in units of $\pi$) and `ndofs`.

Example:

```python
proj_cell = proj.cols[col_key].cells[cell_key]
[th0, th1] = proj_cell.pos
u_ija = proj_cell.vals[i, j, a]
```
