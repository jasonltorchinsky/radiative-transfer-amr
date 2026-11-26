`cell`
================================================================================

This subdirectory contains the source code regarding angular *hp*-elements (`cells`) in an *hp*-mesh: the `Cell` class and the cell key.

How It Fits Into the Whole
--------------------------------------------------------------------------------

A cell is an angular *h*-element $K$ (Section 2.1 of the publication): an arc $[\theta_0, \theta_1] \subseteq [0, 2\pi]$ with $n^K_\theta$ Legendre-Gauss nodes. Each `dg.mesh.column.Column` holds its own angular mesh in `Column.cells`, keyed by cell key, and a spatial-angular element is a pair (column, cell).

- `dg.mesh` creates cells in `Mesh.__init__` (root cell), `ref_cell_h` (children of an angularly *h*-refined cell), `from_file` and `remove_th`; `ref_col_spt_h` deep-copies the cells of a column into its children.
- `dg.mesh.ref_cell`, `dg.mesh.nhbr_cells_in_nhbr_col`, `dg.mesh.nhbr_cells_spt` and `Column.nhbr_cell` compute cell keys with `calc_key`.
- `rt`, `dg.matrix`, `dg.projection` and `amr` read `Cell.pos`, `Cell.ndofs` and `Cell.quadrant`; `dg.projection.Projection_Cell` copies the attributes of a cell.

This subdirectory calls only `numpy` (for `np.pi` in `Cell.__str__`). It does not use MPI.

Conventions
--------------------------------------------------------------------------------

See also `dg/mesh/README.md` (Data Model and Conventions).

- `lv` $= l$ : the angular refinement level, independent of the level of the column. The root cell has level 0 and covers $[0, 2\pi]$.
- `idx` : `int` in $[0, 2^l)$; the cell covers $[2\pi \, idx/2^l, 2\pi (idx + 1)/2^l]$, and `pos = [th0, th1]` holds these endpoints in radians. The children of a cell have indices `2 * idx` (lower half) and `2 * idx + 1` (upper half).
- `key = calc_key(idx, lv)` $= 2^l - 1 + idx$ : binary-heap numbering. The root cell has key 0 and the children of the cell with key $k$ have keys $2k + 1$ and $2k + 2$. Keys are local to a column: the same arc has the same key in every column.
- `nhbr_keys = [k_minus, k_plus]` : keys of the cells of the same column across $\theta_0$ and across $\theta_1$, with periodic wrap-around at $0 = 2\pi$. The root cell has `[0, 0]`; the dummy cell of a mesh without angular dimension has `[None, None]`.
- `quadrant` : `0`, `1`, `2` or `3` if the arc lies in the closed quadrant $[q \, \pi/2, (q + 1) \pi/2]$, else `None`. Set by `dg.mesh.ref_cell_h`; the root cell and the level-1 cells have `None`. The solver uses it for upwinding and rejects meshes with angular dimension that have a leaf cell with `quadrant is None`.
- `ndofs = [nth]` : the number of LG nodes in $\theta$; `[1]` for the dummy cell of a mesh without angular dimension (whose `pos` is `[0, 0]`).

Package Initialization
--------------------------------------------------------------------------------

`__init__.py` defines `src_dir` (`str`, the normalized path of the `src` directory), appends it to `sys.path` if absent, and exports:

- `calc_key` : the cell-key function (`calc_key.py`).
- `Cell` : the cell class (`class_Cell.py`).

Source Files
--------------------------------------------------------------------------------

The source files contained in the `cell` subdirectory are:

- `__init__.py` : Adds the `src` directory to `sys.path` and exports `calc_key` and `Cell`.
- `calc_key` : Calculates the key of an angular *hp*-element (`cell`) from its index and level.
- `class_Cell` : The `Cell` class implementing angular *hp*-elements (`cells`).

Functions and Classes
--------------------------------------------------------------------------------

### `calc_key(idx, lv)` (`calc_key.py`)

Returns the unique `int` key of the cell with index `idx` (`int`) at level `lv` (`int`, $l \ge 0$):

$$\mathrm{key} = \sum_{l' = 0}^{l - 1} 2^{l'} + idx = 2^l - 1 + idx.$$

```python
from dg.mesh.column.cell import calc_key

calc_key(0, 0) # 0, the root cell [0, 2 pi]
calc_key(1, 1) # 2, the cell [pi, 2 pi]
calc_key(3, 2) # 6, the cell [3 pi / 2, 2 pi]
```

### `class Cell(pos, idx, lv, is_lf, ndofs, quadrant, nhbr_keys)` (`class_Cell.py`)

An angular *hp*-element. The arguments are stored as attributes without copying, and `key` is computed with `calc_key(idx, lv)`.

- `pos` (`list` of 2 `float`) : `[th0, th1]`, the endpoints of the arc in radians.
- `idx` (`int`) : angular index at level `lv`.
- `lv` (`int`) : angular refinement level.
- `is_lf` (`bool`) : whether the cell is a leaf. Cells stored in a `Column` are always leaves, since *h*-refinement deletes the parent.
- `ndofs` (`list` of 1 `int`) : `[nth]`.
- `quadrant` (`int` or `None`) : angular quadrant (see Conventions).
- `nhbr_keys` (`list` of 2 `int` or `None`) : `[k_minus, k_plus]`.

Attributes: `pos`, `idx`, `lv`, `key` (`int`), `is_lf`, `ndofs`, `quadrant`, `nhbr_keys`.

Methods:

- `__eq__(other)` : `True` if all attributes are equal. It does not know which column a cell belongs to, so equal cells in different columns compare equal.
- `__str__()` : A multi-line summary with `idx`, `lv`, `key`, `pos` (in units of $\pi$), `is_lf`, `ndofs`, `quadrant` and `nhbr_keys`.

```python
import numpy as np
from dg.mesh.column.cell import Cell

cell = Cell(pos = [0., np.pi / 2.], idx = 0, lv = 2, is_lf = True,
            ndofs = [4], quadrant = 0, nhbr_keys = [6, 4])
cell.key # 3
```

Cells are normally created by `dg.mesh.Mesh` and its refinement methods rather than directly.
