`mesh`
================================================================================

This subdirectory contains the source code regarding the spatial-angular *hp*-mesh: the `Mesh` class, which holds the spatial *hp*-elements (`columns`, or `cols`) and, in each column, the angular *hp*-elements (`cells`); the neighbor queries between them; *h*- and *p*-refinement with the 1-irregularity and even-degree criteria; the count of degrees of freedom (DOFs); and reading and writing a mesh as JSON.

Equation and section numbers below refer to the publication (Torchinsky, Du & Stechmann, JQSRT 348 (2026) 109687); numbers prefixed with `SI-` refer to its Supplemental Information.

How It Fits Into the Whole
--------------------------------------------------------------------------------

The mesh is '2 + 1'-dimensional (Section 2.1 of the publication): the rectangle $\Omega = [0, L_x] \times [0, L_y]$ is covered by rectangular columns $C$, and each column has its own angular mesh of arcs $K \subseteq [0, 2\pi]$. A spatial-angular element is a pair (column, cell).

- The experiment scripts (`experiments/*/mesh.py`) build the initial mesh with `Mesh(...)` and refine it uniformly with `Mesh.ref_mesh`.
- `rt.Problem.solve` checks the mesh, counts DOFs with `Mesh.get_ndof`, and assembles the matrices by looping over `Mesh.cols` and `Column.cells` in increasing key order. The boundary propagation matrix uses `Column.nhbr_keys`, `Mesh.nhbr_cells_in_nhbr_col` and `Cell.quadrant`.
- `dg.matrix` orders DOFs by column key and cell key (`get_col_idxs`, `get_cell_idxs`) and uses `Column.nhbr_keys`, `Cell.quadrant` and `Mesh.pbcs` to find the inflow boundary (`get_intr_mask`, `get_bdry_mask`).
- `dg.projection.Projection` deep-copies a `Mesh` and stores nodal values on each (column, cell) pair. `Projection.to_file` calls `Mesh.to_file`, `dg.projection.from_file` calls `mesh.from_file`, and `dg.projection.intg_th` calls `remove_th`.
- `amr.error_indicator` computes error indicators on the mesh (the spatial jump indicator uses `Mesh.nhbr_cells_in_nhbr_col`), then `Error_Indicator.ref_by_ind` deep-copies the mesh and calls `Mesh.ref_cell` and `Mesh.ref_col(kind = "spt")` on the marked elements (steps 5-7 of the algorithm in Section 3.2).
- `tools.dg.mesh` plots meshes and neighbor relations.

This subdirectory calls only `consts` (`consts.PI`) and its own subpackage `column`. It does not use MPI. In the experiments the mesh is refined on the root MPI process (`consts.COMM_ROOT`) and then broadcast with `mpi_comm.bcast`, so every process holds a full copy of the mesh.

Data Model and Conventions
--------------------------------------------------------------------------------

This section documents conventions requested in issue #21.

### Objects

- `Mesh.cols` : `dict` mapping column key (`int`) to `Column`.
- `Column.cells` : `dict` mapping cell key (`int`) to `Cell`. Cell keys are local to the column: columns with the same angular mesh use the same cell keys.

Only leaves are stored. *h*-refinement adds the children and deletes the parent (`del_col`, `del_cell`), so every stored column and cell has `is_lf == True`; the `is_lf` attribute and the `is_lf` checks in the code are kept for completeness.

### Positions

- `Column.pos = [x0, y0, x1, y1]` : the column covers $[x_0, x_1] \times [y_0, y_1]$. The root column has `pos = [0, 0, Ls[0], Ls[1]]`.
- `Cell.pos = [th0, th1]` : the cell covers the arc $[\theta_0, \theta_1]$, in radians, with $0 \le \theta_0 < \theta_1 \le 2\pi$. The root cell has `pos = [0, 2 * consts.PI]`. On a mesh without angular dimension (`has_th == False`) each column has one dummy cell with `pos = [0, 0]`.

### Levels and indices

- `Column.lv` : the spatial refinement level; the root column has level 0, and each spatial *h*-refinement adds 1. A column at level $l$ has side lengths $L_x / 2^l$ and $L_y / 2^l$.
- `Column.idx = [iy, ix]` : the position of the column in the uniform $2^l \times 2^l$ grid of level $l$, stored as **[y index, x index]**. The column covers $x \in [ix \, L_x / 2^l, (ix + 1) L_x / 2^l]$ and $y \in [iy \, L_y / 2^l, (iy + 1) L_y / 2^l]$. `ref_col_spt_h` gives the children of a column with `idx = [i, j]` the indices `[2i, 2j]` (bottom-left), `[2i+1, 2j]` (top-left), `[2i, 2j+1]` (bottom-right) and `[2i+1, 2j+1]` (top-right).
- `Cell.lv` : the angular refinement level, independent of the column level. A cell at level $l$ has length $2\pi / 2^l$.
- `Cell.idx` : `int` in $[0, 2^l)$; the cell covers $[2\pi \, idx / 2^l, 2\pi (idx + 1) / 2^l]$. The children of a cell with index `idx` have indices `2 * idx` (lower half) and `2 * idx + 1` (upper half).

### Keys

- Column key (`column.calc_key`) : $\mathrm{key} = \sum_{l' = 0}^{l - 1} 4^{l'} + 2^l \, ix + iy = (4^l - 1)/3 + 2^l \cdot$ `idx[1]` $+$ `idx[0]`. The root column has key 0; the level-1 columns have keys 1 (bottom-left), 2 (top-left), 3 (bottom-right), 4 (top-right).
- Cell key (`column.cell.calc_key`) : $\mathrm{key} = \sum_{l' = 0}^{l - 1} 2^{l'} + idx = 2^l - 1 + idx$. This is binary-heap numbering: the root cell has key 0, and the children of the cell with key $k$ have keys $2k + 1$ and $2k + 2$.

Keys are unique over all levels. Sorting by key orders elements by level first; within a level, columns are ordered by x index and then by y index, and cells by increasing $\theta$. The global DOF vector takes leaf columns in increasing column-key order and, within a column, leaf cells in increasing cell-key order (see `dg/projection/README.md`, Global DOF Ordering).

### Column neighbors

`Column.nhbr_keys` is a list of four faces, numbered counterclockwise from the right face:

| `F` | Face | Neighbor direction |
| --- | --- | --- |
| 0 | right, $x = x_1$ | $+x$ |
| 1 | top, $y = y_1$ | $+y$ |
| 2 | left, $x = x_0$ | $-x$ |
| 3 | bottom, $y = y_0$ | $-y$ |

`nhbr_keys[F] = [k0, k1]` holds the keys of the columns across face `F`, in counterclockwise order around the column: bottom then top on the right face, right then left on the top face, top then bottom on the left face, left then right on the bottom face.

- A neighbor at the same level or one level coarser: `k0 == k1`, the key of that neighbor.
- Two neighbors one level finer: `k0 != k1`, the keys of the two finer columns in the order above.
- The face lies on a non-periodic boundary: `[None, None]`.
- Periodic boundary (`pbcs[0]` for x, `pbcs[1]` for y): the neighbor is the column across the opposite side of $\Omega$. The root column of a periodic direction is its own neighbor (`[0, 0]`).

`Mesh.nhbr_cells_in_nhbr_col` returns the cells of a neighboring column that overlap a given cell in angle.

### Cell neighbors

`Cell.nhbr_keys = [k_minus, k_plus]` holds the keys of the cells in the same column that share the endpoint $\theta_0$ (`k_minus`) and $\theta_1$ (`k_plus`). The angle is periodic, so the cell starting at $\theta = 0$ has the cell ending at $2\pi$ as `k_minus`. In a 1-irregular angular mesh each endpoint has exactly one neighbor. The root cell is its own neighbor (`[0, 0]`). Dummy cells of a mesh without angular dimension have `[None, None]`.

### Quadrants

`Cell.quadrant` is `SS` in `0, 1, 2, 3` if the cell lies in the closed quadrant $[SS \, \pi/2, (SS + 1) \, \pi/2]$, and `None` otherwise. It is set when a cell is created by angular *h*-refinement (`ref_cell_h`): children of a cell with a quadrant inherit it; otherwise each child is tested against the four quadrants. The root cell and the two level-1 cells have `None`.

The solver uses the quadrant to decide which spatial faces of a column are outflow faces for the cell (`rt/boundary_convection_matrix.py`; e.g. quadrant 0 has outflow through faces 0 and 1) and which boundary DOFs are inflow DOFs (`dg/matrix/get_masks.py`). `rt.Problem.solve` therefore rejects a mesh with angular dimension that has a leaf cell with `quadrant is None`; refine the mesh in angle at least twice. The publication does not state this restriction.

### Degrees of freedom

`Column.ndofs = [nx, ny]` and `Cell.ndofs = [nth]` are the numbers of nodes $n^C_x$, $n^C_y$ (Legendre-Gauss-Lobatto) and $n^K_\theta$ (Legendre-Gauss) of Eq. 5; the polynomial degree in each dimension is one less. A spatial-angular element (column, cell) has $n_x n_y n_\theta$ DOFs. On a mesh without angular dimension every cell has `ndofs = [1]`.

Refinement Rules
--------------------------------------------------------------------------------

Refinement is done through `Mesh.ref_mesh`, `Mesh.ref_col`, `Mesh.ref_col_spt` and `Mesh.ref_cell`. Each operates on leaves and recursively refines neighbors first or afterwards so that the following invariants hold (Section 3.2):

- **Spatial *h*-refinement** (`ref_col_spt_h`) : a column is split into four children of half the side length. The children copy the parent's `ndofs` and a deep copy of its angular mesh (`cells`). Before the split, every neighbor that is one level coarser is spatially *h*-refined (`ref_col(nhbr_key, kind = "spt")`, recursively), so neighboring columns differ by at most one level (spatial 1-irregularity, at most two neighbors per face). Afterwards the neighbors' `nhbr_keys` are updated and the parent is deleted.
- **Angular *h*-refinement** (`ref_cell_h`) : a cell is split into two halves with the parent's `ndofs`. Before the split, each angular neighbor in the same column and each overlapping cell in a spatially neighboring column (`nhbr_cells_in_nhbr_col`) that is one angular level coarser is angularly *h*-refined first (Fig. 2). This keeps the angular 1-irregularity within a column and between neighboring columns.
- **Spatial *p*-refinement** (`ref_col_spt_p`) : `nx` and `ny` of the column are increased by 1 (for all cells of the column at once, as in Section 3.2).
- **Angular *p*-refinement** (`ref_cell_p`) : `nth` of the cell is increased by 1.
- **DOF difference of at most one** : after a *p*-refinement, every neighbor (spatial neighbors for columns; overlapping cells in neighboring columns and the two angular neighbors for cells) whose number of DOFs differs by more than one is *p*-refined, recursively. Section 3.2 of the publication does not state this rule.
- **Even-degree criterion** (Section 3.2, enforced in `ref_col_spt_p` and `ref_cell_p`): if an element has several neighbors through one face, they must have the same number of DOFs. After *p*-refining element $E$, the code goes to each neighbor $N$ of $E$, looks at the face of $N$ that $E$ lies behind, and *p*-refines every element behind that face until it has at least as many DOFs as $E$ (`while ... < ...`). For columns both `nx` and `ny` are compared; for cells, `nth`. The code raises the other elements to match and never lowers $E$, so it keeps the criterion if it held before the refinement. `tests/test_dg/test_mesh/test_even_degree.py` checks the criterion after spatial, angular and random *p*-refinements.

*h*-refinement does not change `ndofs`, so it keeps the even-degree criterion. Spatial refinement does not change angular meshes, so it keeps angular 1-irregularity between columns.

Mesh File Format
--------------------------------------------------------------------------------

`Mesh.to_file(file_path)` writes the mesh with `json.dump`, and `from_file(file_path)` reads it back (`from_file(mesh.to_file(...)) == mesh`). The file is one JSON object:

```
{
  "Ls": [Lx, Ly],
  "pbcs": [periodic_in_x, periodic_in_y],
  "has_th": true | false,
  "cols": {
    "<column key>": {
      "pos": [x0, y0, x1, y1],
      "idx": [iy, ix],
      "lv": <int>,
      "key": <int>,
      "is_lf": true,
      "ndofs": [nx, ny],
      "nhbr_keys": [[k0, k1], [k0, k1], [k0, k1], [k0, k1]],
      "cells": {
        "<cell key>": {
          "pos": [th0, th1],
          "idx": <int>,
          "lv": <int>,
          "key": <int>,
          "is_lf": true,
          "ndofs": [nth],
          "quadrant": 0 | 1 | 2 | 3 | null,
          "nhbr_keys": [k_minus, k_plus]
        }, ...
      }
    }, ...
  }
}
```

- Column and cell keys are JSON object keys and therefore strings; `None` is written as `null`. Columns and cells appear in dictionary insertion order, not key order; sort them with `sorted(keys, key = int)` to recover the DOF order.
- `"key"` is redundant: `from_file` ignores it and recomputes each key from `idx` and `lv`.
- Example: `Mesh(Ls = [2.0, 1.0], pbcs = [True, False], ndofs = [3, 3, 4], has_th = True)` followed by `ref_mesh(kind = "ang")` is written as

```
{"Ls": [2.0, 1.0], "pbcs": [true, false], "has_th": true, "cols": {"0": {"pos": [0, 0, 2.0, 1.0], "idx": [0, 0], "lv": 0, "key": 0, "is_lf": true, "ndofs": [3, 3], "nhbr_keys": [[0, 0], [null, null], [0, 0], [null, null]], "cells": {"1": {"pos": [0, 3.141592653589793], "idx": 0, "lv": 1, "key": 1, "is_lf": true, "ndofs": [4], "quadrant": null, "nhbr_keys": [2, 2]}, "2": {"pos": [3.141592653589793, 6.283185307179586], "idx": 1, "lv": 1, "key": 2, "is_lf": true, "ndofs": [4], "quadrant": null, "nhbr_keys": [1, 1]}}}}}
```

The experiments write the mesh of each trial as `mesh.json` (and the high-resolution mesh as `mesh_hr.json`) through `Projection.to_file`.

Package Initialization
--------------------------------------------------------------------------------

`__init__.py` defines `src_dir` (`str`, the normalized path of the `src` directory), appends it to `sys.path` if absent, and exports:

- `column` : the subpackage `dg.mesh.column` (see `column/README.md`).
- `Mesh` : the mesh class (`class_Mesh.py`).
- `from_file` : reads a mesh from a JSON file (`from_file.py`).
- `remove_th` : returns a copy of a mesh without angular dimension (`remove_th.py`).

The other modules define methods that `class_Mesh.py` binds to `Mesh` by importing them in the class body, plus the helper functions `ref_cell_p`, `ref_cell_h`, `ref_col_spt_p` and `ref_col_spt_h`, which are not exported.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `mesh` subdirectory are:

- `__init__.py` : Adds the `src` directory to `sys.path` and exports `column`, `Mesh`, `from_file` and `remove_th`.
- `add_col` : Method `Mesh.add_col`, which adds a spatial *hp*-element (`column`, or `col`) to the *hp*-mesh.
- `class_Mesh` : The `Mesh` class.
- `del_col` : Method `Mesh.del_col`, which deletes a spatial *hp*-element (`column`, or `col`) from the *hp*-mesh.
- `from_file` : Reads an *hp*-mesh from a JSON file.
- `get_ndof` : Method `Mesh.get_ndof`, which counts the degrees of freedom in the *hp*-mesh.
- `nhbr_cells_in_nhbr_col` : Method `Mesh.nhbr_cells_in_nhbr_col`, which returns the keys of the angular *hp*-elements (`cells`) in a neighboring spatial *hp*-element (`column`, or `col`) that overlap a given cell in angle.
- `nhbr_cells_spt` : Method `Mesh.nhbr_cells_spt`, which returns the neighboring columns of a column along one axis and the overlapping cells in each. Not used by the solver.
- `nhbr_cols` : Method `Mesh.nhbr_cols`, which finds the neighboring columns of a column from its index. Not used by the solver.
- `ref_cell` : Method `Mesh.ref_cell` and helpers `ref_cell_p`, `ref_cell_h`, which *p*- or *h*-refine an angular *hp*-element (`cell`) while keeping the 1-irregularity and even-degree criteria.
- `ref_col_ang` : Method `Mesh.ref_col_ang`, which angularly *h*-refines every cell of a column. Not used by the solver.
- `ref_col_spt` : Method `Mesh.ref_col_spt` and helpers `ref_col_spt_p`, `ref_col_spt_h`, which spatially *p*- or *h*-refine a column while keeping the 1-irregularity and even-degree criteria.
- `ref_col` : Method `Mesh.ref_col`, which refines a column in angle, space, or both.
- `ref_mesh` : Method `Mesh.ref_mesh`, which refines every element of the *hp*-mesh.
- `remove_th` : Returns a copy of the *hp*-mesh without the angular dimension.
- `to_file` : Method `Mesh.to_file`, which writes the *hp*-mesh to a JSON file.

Functions and Classes
--------------------------------------------------------------------------------

### `class Mesh(Ls, pbcs, ndofs = [2, 2, 2], has_th = False)` (`class_Mesh.py`)

Creates a mesh with one column (key 0, level 0, covering $\Omega$) that has one cell (key 0, level 0).

- `Ls` (`list` of 2 `float`) : `[Lx, Ly]`, the side lengths of $\Omega = [0, L_x] \times [0, L_y]$.
- `pbcs` (`list` of 2 `bool`) : `[periodic_in_x, periodic_in_y]`. In a periodic direction the root column is its own neighbor across the two faces of that direction; otherwise those faces get `[None, None]`.
- `ndofs` (`list` of 3 `int`, default `[2, 2, 2]`) : `[nx, ny, nth]`, the numbers of nodes of the root column (`ndofs[0:2]`) and root cell (`[ndofs[2]]`). The solver needs `nx, ny >= 2`.
- `has_th` (`bool`, default `False`) : If `True`, the root cell covers $[0, 2\pi]$ with `ndofs = [nth]` and `nhbr_keys = [0, 0]`. If `False`, the root cell is a dummy with `pos = [0, 0]`, `ndofs = [1]` and `nhbr_keys = [None, None]`, and `ndofs[2]` is ignored.

Attributes:

- `Ls` (`list`), `pbcs` (`list`), `has_th` (`bool`) : The arguments, stored without copying.
- `cols` (`dict`) : Column key (`int`) to `Column`.

Methods (each documented below): `to_file`, `add_col`, `del_col`, `ref_col`, `nhbr_cols`, `nhbr_cells_in_nhbr_col`, `nhbr_cells_spt`, `ref_col_spt`, `ref_col_ang`, `ref_cell`, `ref_mesh`, `get_ndof`, and

- `__eq__(other)` : `True` if `Ls`, `pbcs`, `has_th` and `cols` are equal (`Column.__eq__`, which compares all attributes and the cells).
- `__str__()` : Lists `Ls`, `pbcs`, `has_th` and the sorted column keys.

Example (as in `experiments/*/mesh.py`):

```python
from dg.mesh import Mesh

mesh = Mesh(Ls = [3.0, 2.0], pbcs = [False, False], ndofs = [3, 3, 4], has_th = True)
for _ in range(2):
    mesh.ref_mesh(kind = "ang", form = "h") # 4 cells per column, one per quadrant
for _ in range(2):
    mesh.ref_mesh(kind = "spt", form = "h") # 4 x 4 columns
print(mesh.get_ndof())                      # 16 * 3 * 3 * 4 * 4 = 2304
```

### `Mesh.add_col(col)` (`add_col.py`)

Inserts `col` (`Column`) into `self.cols` under `col.key`, replacing any column with that key. Does not set neighbor keys. Returns `None`.

### `Mesh.del_col(col_key)` (`del_col.py`)

Removes the column with key `col_key` (`int`) from `self.cols`; raises `KeyError` if absent. Does not update neighbor keys. Returns `None`.

### `Mesh.get_ndof()` (`get_ndof.py`)

Returns the total number of DOFs (`int`), $\sum_C n^C_x n^C_y \sum_{K \in C} n^K_\theta$ over leaf columns and leaf cells. This is the length of the global DOF vector. For a mesh without angular dimension it is $\sum_C n^C_x n^C_y$.

### `Mesh.to_file(file_path = "mesh.json")` (`to_file.py`)

Writes the mesh to `file_path` (`str`) as JSON (see Mesh File Format), overwriting the file. Returns `None`. It writes from the process that calls it; the experiments call it (through `Projection.to_file`) on the root process only.

### `from_file(file_path = "mesh.json")` (`from_file.py`)

Reads a mesh written by `Mesh.to_file` and returns it as a new `Mesh`. It creates `Mesh(Ls, pbcs, has_th = has_th)`, deletes its root column, and adds one `Column` per entry of `"cols"`, each with one `Cell` per entry of `"cells"` (cell keys are converted to `int`; column and cell keys are recomputed from `idx` and `lv`). The file is not validated.

```python
from dg.mesh import from_file

mesh = from_file("out/trial/mesh.json")
```

### `remove_th(mesh)` (`remove_th.py`)

Returns a new `Mesh` with the same `Ls` and `pbcs`, `has_th = False`, and a copy of each leaf column of `mesh` (same `pos`, `idx`, `lv`, `ndofs`, `nhbr_keys` and key) whose angular mesh is replaced by one dummy cell (key 0, `pos = [0, 0]`, `ndofs = [1]`, `quadrant = None`, `nhbr_keys = [None, None]`). Used by `dg.projection.intg_th` to hold the angular integral of a `Projection`.

- `mesh` (`Mesh`) : The input mesh; not modified.

Side effect to be aware of: `nhbr_keys` is copied with `[:]`, so the outer list is new but the four inner `[k0, k1]` lists are shared with `mesh`. Changing them in place in one mesh changes the other.

### `Mesh.ref_mesh(kind = "all", form = "h")` (`ref_mesh.py`)

Refines every column present when the call starts, in increasing key order, by calling `self.ref_col(col_key, kind = ..., form = form)`.

- `kind` (`str`, default `"all"`) : `"ang"` refines every cell of every column, `"spt"` every column in space, `"all"` first all cells and then all columns. Other values do nothing.
- `form` (`str`, default `"h"`) : `"h"`, `"p"` or `"hp"`, passed on to `ref_col`.

Modifies the mesh in place; returns `None`. On a mesh where all columns have the same level and all cells the same level, this is a uniform refinement. Otherwise the recursive 1-irregularity and DOF-difference rules can refine some elements more than once in one call.

### `Mesh.ref_col(col_key, kind = "all", form = "h")` (`ref_col.py`)

Refines the column `col_key` (`int`), which must be a leaf (`AssertionError` otherwise).

- `kind` (`str`, default `"all"`) : `"ang"` calls `ref_cell(col_key, cell_key, form = form)` for every cell of the column (keys taken before the loop, in increasing order); `"spt"` calls `ref_col_spt(col_key, form = form)`; `"all"` does both, angular first. Any other value raises `ValueError`.
- `form` (`str`, default `"h"`) : `"h"`, `"p"` or `"hp"`.

Modifies the mesh in place (spatial *h*-refinement deletes `col_key`); returns `None`.

### `Mesh.ref_col_spt(col_key, form = "h")` (`ref_col_spt.py`)

Spatially refines the leaf column `col_key` (`int`). `form = "p"` calls `ref_col_spt_p`, `"h"` calls `ref_col_spt_h`, `"hp"` calls `ref_col_spt_p` and then `ref_col_spt_h` (so the four children get the raised `ndofs`). Any other value raises `ValueError`. Returns `None`.

#### `ref_col_spt_p(mesh, col_key)` (`ref_col_spt.py`, not exported)

Spatial *p*-refinement: increases `ndofs[0]` and `ndofs[1]` of column `col_key` by 1. Then, recursively:

1. every neighbor whose `ndofs[0]` or `ndofs[1]` differs from the column's by more than 1 is *p*-refined;
2. even-degree criterion: for each neighbor $N$ across face `F`, every column across face `(F + 2) % 4` of $N$ is *p*-refined until both its `ndofs` are at least those of the column.

Asserts that the column and all neighbors visited are leaves. Modifies the mesh in place; returns `None`.

#### `ref_col_spt_h(mesh, col_key)` (`ref_col_spt.py`, not exported)

Spatial *h*-refinement of column `col_key`:

1. Every neighbor one level coarser is spatially *h*-refined first (`mesh.ref_col(nhbr_key, kind = "spt")`, which recursively applies this rule).
2. Four children are added with `idx`, `pos` and keys as in Data Model and Conventions, level `lv + 1`, a copy of `ndofs`, a `copy.deepcopy` of `cells`, and neighbor keys built from the parent's `nhbr_keys`. A column that was its own periodic neighbor makes its children neighbors of each other.
3. The `nhbr_keys` of the parent's neighbors are updated: a neighbor at the same level gets the adjacent child in the matching entry of its opposite face; a neighbor one level finer gets the adjacent child in both entries.
4. The parent is deleted.

Modifies the mesh in place; returns `None`.

### `Mesh.ref_cell(col_key, cell_key, form = "h")` (`ref_cell.py`)

Angularly refines leaf cell `cell_key` (`int`) of leaf column `col_key` (`int`). `form = "p"` calls `ref_cell_p`, `"h"` calls `ref_cell_h`, `"hp"` calls `ref_cell_p` and then `ref_cell_h`. Any other value raises `ValueError`. Returns `None`.

#### `ref_cell_p(mesh, col_key, cell_key)` (`ref_cell.py`, not exported)

Angular *p*-refinement: increases `ndofs[0]` of the cell by 1. Then, recursively:

1. every cell in a spatially neighboring column that overlaps the cell in angle (`nhbr_cells_in_nhbr_col`) and whose `nth` differs by more than 1 is *p*-refined;
2. each of the two angular neighbors in the same column (`Column.nhbr_cell`) whose `nth` differs by more than 1 is *p*-refined;
3. even-degree criterion: for each overlapping cell $N$ in a neighboring column across face `F`, every cell that overlaps $N$ in a column across face `(F + 2) % 4` of $N$'s column is *p*-refined until its `nth` is at least the cell's.

Asserts that the visited columns and cells are leaves. Modifies the mesh in place; returns `None`.

#### `ref_cell_h(mesh, col_key, cell_key)` (`ref_cell.py`, not exported)

Angular *h*-refinement of a cell with index `idx`, level `lv` and `pos = [th0, th1]`:

1. Each angular neighbor (`cell.nhbr_keys`) and each overlapping cell in a spatially neighboring column with level `lv - 1` is angularly *h*-refined first (`mesh.ref_cell(...)`, recursively).
2. Two children are added to the column: indices `2 * idx` and `2 * idx + 1`, level `lv + 1`, `pos` `[th0, th_mid]` and `[th_mid, th1]`, a copy of `ndofs`, and `quadrant` as described in Quadrants. The children are each other's neighbors; the outer neighbors are the parent's (for the root cell, which is its own neighbor, the two children become each other's neighbors on both sides). Raises `RuntimeError` if an outer neighbor is not a leaf.
3. The outer neighbors' `nhbr_keys` are updated to point at the adjacent child, and the parent is deleted.

Modifies the mesh in place; returns `None`.

### `Mesh.ref_col_ang(col_key)` (`ref_col_ang.py`)

Angularly *h*-refines every cell of leaf column `col_key` (`int`), in increasing key order, with `ref_cell(col_key, cell_key)`. Equivalent to `ref_col(col_key, kind = "ang", form = "h")`. Not called anywhere in the package. Returns `None`.

### `Mesh.nhbr_cells_in_nhbr_col(col_key, cell_key, nhbr_col_key)` (`nhbr_cells_in_nhbr_col.py`)

Returns the keys of the leaf cells of column `nhbr_col_key` that overlap cell `cell_key` of column `col_key` in angle, as a `list` `[k0, k1]`:

- the same cell key exists in the neighbor column: `[cell_key, None]`;
- otherwise, the parent of the cell exists there: `[parent_key, None]`;
- otherwise, the two children exist there: `[child_0_key, child_1_key]` (lower half first);
- nothing found: `[None, None]`.

All arguments are `int`. The search uses only key arithmetic, which is valid because all columns index cells the same way and neighboring columns differ by at most one angular level. The two columns need not actually be neighbors. Used by `ref_cell`, `rt.boundary_convection_matrix`, `amr.error_indicator.error.error_spatial_jump` and `tools.dg.mesh.plot_nhbrs`.

```python
for nhbr_col_key in set(mesh.cols[col_key].nhbr_keys[0]) - {None}:  # right face
    [k0, k1] = mesh.nhbr_cells_in_nhbr_col(col_key, cell_key, nhbr_col_key)
```

### `Mesh.nhbr_cols(col_key, axis = 0, nhbr_loc = "+")` (`nhbr_cols.py`)

Finds the neighboring leaf column(s) of column `col_key` (`int`) by index arithmetic, without using `nhbr_keys`. Steps `idx[axis]` by `+1` (`nhbr_loc = "+"`) or `-1` (`"-"`), wraps it modulo $2^l$ if `self.pbcs[axis]`, and looks for a leaf at the same level, then the parent level (only if `lv >= 2`), then the two children adjacent to the column. Returns a `list` `[flag, nhbr_1, nhbr_2]` of a `str` and two `Column` or `None`:

- `"nn"` : no neighbor (index outside the grid, or nothing found); `[None, None]`.
- `"f0"` : one neighbor at the same level in `nhbr_1`.
- `"pm"` / `"pp"` : the neighbor is one level coarser (`nhbr_1`, the parent of the same-level index); `"pm"` if the same-level index along `axis` is even, `"pp"` if odd.
- `"cc"` : two neighbors one level finer in `nhbr_1` and `nhbr_2`.

Since `idx` is `[y index, x index]`, `axis = 0` steps in $y$ and `axis = 1` steps in $x$, while `self.pbcs[axis]` is the periodicity of $x$ for `axis = 0` and of $y$ for `axis = 1`. On a mesh periodic in exactly one direction the result is therefore wrong for columns at the boundary (issue #21). On meshes periodic in both or neither direction it returns the $y$-neighbors for `axis = 0` and the $x$-neighbors for `axis = 1`. The solver uses `Column.nhbr_keys` instead; this method is called only by `nhbr_cells_spt`.

### `Mesh.nhbr_cells_spt(col_key, cell_key, axis = 0, nhbr_loc = "+")` (`nhbr_cells_spt.py`)

Calls `self.nhbr_cols(col_key, axis, nhbr_loc)` (with the axis convention described there) and returns `[[nhbr_col_1, [cell_a, cell_b]], [nhbr_col_2, [cell_a, cell_b]]]`, where each inner pair holds the `Cell` objects (not keys) of that neighbor column that overlap cell `cell_key` in angle, found as in `nhbr_cells_in_nhbr_col` (same key, else parent, else the two children); missing entries are `None`. If there is no neighbor, all entries are `None`. Not used by the solver.

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `mesh` subdirectory is

```
mesh
|
|--- column : Contains source code implementing the Column class (spatial hp-elements) and the column key.
     |
     |--- cell : Contains source code implementing the Cell class (angular hp-elements) and the cell key.
```
