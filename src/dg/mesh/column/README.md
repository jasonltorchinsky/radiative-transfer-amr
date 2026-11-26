`column`
================================================================================

This subdirectory contains the source code regarding spatial *hp*-elements (`columns`, or `cols`) in an *hp*-mesh: the `Column` class, the column key, and the methods that add, delete and find angular *hp*-elements (`cells`) in the angular *hp*-mesh of a column.

How It Fits Into the Whole
--------------------------------------------------------------------------------

A column is a spatial *h*-element $C$ (Section 2.1 of the publication): a rectangle $[x_0, x_1] \times [y_0, y_1]$ with $n^C_x \times n^C_y$ Legendre-Gauss-Lobatto nodes, which carries its own angular mesh (the `cells`). `dg.mesh.Mesh` stores its columns in `Mesh.cols`, keyed by column key.

- `dg.mesh` creates columns in `Mesh.__init__` (root column), `ref_col_spt_h` (children of a spatially *h*-refined column), `from_file` and `remove_th`, and calls `Column.add_cell`, `Column.del_cell` and `Column.nhbr_cell` during angular refinement (`ref_cell`).
- `rt`, `dg.matrix`, `dg.projection` and `amr` read `Column.pos`, `Column.ndofs`, `Column.nhbr_keys` and `Column.cells`; `dg.projection.Projection_Column` copies the attributes of a column.
- `dg.mesh.ref_col_spt` and `dg.mesh.nhbr_cols` compute column keys with `calc_key`.

This subdirectory calls only its own subpackage `cell`. It does not use MPI.

Conventions
--------------------------------------------------------------------------------

These are described in full in `dg/mesh/README.md` (Data Model and Conventions). In short:

- `pos = [x0, y0, x1, y1]`.
- `idx = [iy, ix]` is stored as **[y index, x index]** in the $2^l \times 2^l$ grid of level `lv` $= l$: the column covers $x \in [ix \, L_x/2^l, (ix + 1) L_x/2^l]$ and $y \in [iy \, L_y/2^l, (iy + 1) L_y/2^l]$.
- `key = calc_key(idx, lv)` $= (4^l - 1)/3 + 2^l \cdot$ `idx[1]` $+$ `idx[0]`.
- `nhbr_keys[F] = [k0, k1]` for the faces `F = 0` (right, $+x$), `1` (top, $+y$), `2` (left, $-x$), `3` (bottom, $-y$); the two entries are equal for a neighbor at the same or a coarser level, distinct for two finer neighbors (in counterclockwise order around the column), and `[None, None]` on a non-periodic boundary.
- `ndofs = [nx, ny]`.

Package Initialization
--------------------------------------------------------------------------------

`__init__.py` defines `src_dir` (`str`, the normalized path of the `src` directory), appends it to `sys.path` if absent, and exports:

- `cell` : the subpackage `dg.mesh.column.cell` (see `cell/README.md`).
- `calc_key` : the column-key function (`calc_key.py`).
- `Column` : the column class (`class_Column.py`).

`add_cell`, `del_cell` and `nhbr_cell` are bound to `Column` as methods by importing them in the class body.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `column` subdirectory are:

- `__init__.py` : Adds the `src` directory to `sys.path` and exports `cell`, `calc_key` and `Column`.
- `add_cell` : Method `Column.add_cell`, which adds an angular *hp*-element (`cell`) to the angular *hp*-mesh of the spatial *hp*-element (`column`, or `col`).
- `calc_key` : Calculates the key of a spatial *hp*-element (`column`, or `col`) from its index and level.
- `class_Column` : The `Column` class implementing spatial *hp*-elements (`columns`, or `cols`) in an *hp*-mesh.
- `del_cell` : Method `Column.del_cell`, which deletes an angular *hp*-element (`cell`) from the angular *hp*-mesh of the spatial *hp*-element (`column`, or `col`).
- `nhbr_cell` : Method `Column.nhbr_cell`, which returns the neighboring angular *hp*-element (`cell`) of a cell, in angle, in the angular *hp*-mesh of the spatial *hp*-element (`column`, or `col`).

Functions and Classes
--------------------------------------------------------------------------------

### `calc_key(idx, lv)` (`calc_key.py`)

Returns the unique `int` key of the column with index `idx` (`list` of 2 `int`, `[iy, ix]`) at level `lv` (`int`, $l \ge 0$):

$$\mathrm{key} = \sum_{l' = 0}^{l - 1} 4^{l'} + 2^l \, \mathtt{idx[1]} + \mathtt{idx[0]}.$$

The first term counts all columns of the coarser levels, so keys are unique across levels and sorting by key orders columns by level, then by x index, then by y index.

```python
from dg.mesh.column import calc_key

calc_key([0, 0], 0) # 0, the root column
calc_key([1, 0], 1) # 2, the top-left column of level 1 (y index 1, x index 0)
calc_key([0, 1], 1) # 3, the bottom-right column of level 1
```

### `class Column(pos, idx, lv, is_lf, ndofs, cells, nhbr_keys)` (`class_Column.py`)

A spatial *hp*-element. The arguments are stored as attributes without copying, and `key` is computed with `calc_key(idx, lv)`.

- `pos` (`list` of 4 `float`) : `[x0, y0, x1, y1]`, the corners of the column.
- `idx` (`list` of 2 `int`) : `[iy, ix]`, the y and x indices at level `lv`.
- `lv` (`int`) : spatial refinement level.
- `is_lf` (`bool`) : whether the column is a leaf. Columns stored in a `Mesh` are always leaves, since *h*-refinement deletes the parent.
- `ndofs` (`list` of 2 `int`) : `[nx, ny]`, the numbers of LGL nodes in $x$ and $y$.
- `cells` (`dict`) : cell key (`int`) to `Cell`, the angular mesh of the column.
- `nhbr_keys` (`list` of 4 `list` of 2 `int` or `None`) : neighbor column keys through each face (see Conventions).

Attributes: `pos`, `idx`, `lv`, `key` (`int`), `is_lf`, `ndofs`, `cells`, `nhbr_keys`.

Methods:

- `add_cell(cell)`, `del_cell(cell_key)`, `nhbr_cell(cell_key, nhbr_loc = "+")` : Documented below.
- `__eq__(other)` : `True` if all attributes are equal, including the `cells` dictionaries (compared with `Cell.__eq__`).
- `__str__()` : A multi-line summary with `idx`, `lv`, `key`, `pos`, `is_lf`, `ndofs`, the list of cell keys and `nhbr_keys`.

```python
from dg.mesh.column import Column
from dg.mesh.column.cell import Cell

cell = Cell(pos = [0., 6.283185307179586], idx = 0, lv = 0, is_lf = True,
            ndofs = [4], quadrant = None, nhbr_keys = [0, 0])
col = Column(pos = [0., 0., 1., 1.], idx = [0, 0], lv = 0, is_lf = True,
             ndofs = [3, 3], cells = {0 : cell},
             nhbr_keys = [[None, None], [None, None], [None, None], [None, None]])
```

Columns are normally created by `dg.mesh.Mesh` and its refinement methods rather than directly.

### `Column.add_cell(cell)` (`add_cell.py`)

Inserts `cell` (`Cell`) into `self.cells` under `cell.key`, replacing any cell with that key. Does not update neighbor keys. Returns `None`.

### `Column.del_cell(cell_key)` (`del_cell.py`)

Removes the cell with key `cell_key` (`int`) from `self.cells`; raises `KeyError` if absent. Does not update neighbor keys. Returns `None`.

### `Column.nhbr_cell(cell_key, nhbr_loc = "+")` (`nhbr_cell.py`)

Returns the leaf `Cell` of this column that neighbors cell `cell_key` (`int`) in angle, found by index arithmetic (it does not use `Cell.nhbr_keys`).

- `nhbr_loc` (`str`, default `"+"`) : `"+"` for the neighbor across the upper endpoint $\theta_1$, `"-"` for the neighbor across $\theta_0$. Any other value leaves the index unchanged, so the cell itself is returned.

The neighbor index at the same level is `idx + 1` (`"+"`) or `idx - 1` (`"-"`) modulo $2^l$ (the angle is periodic). The method returns the leaf with that index at the same level, else its parent at level $l - 1$, else the adjacent child at level $l + 1$ (index `2 * nhbr_idx` for `"+"`, `2 * nhbr_idx + 1` for `"-"`). It returns `None` if none of these is a leaf, which does not happen in a 1-irregular angular mesh. Used by `dg.mesh.ref_cell_p`.

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `column` subdirectory is

```
column
|
|--- cell : Contains source code implementing the Cell class (angular hp-elements) and the cell key.
```
