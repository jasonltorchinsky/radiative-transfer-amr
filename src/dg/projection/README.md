`projection`
================================================================================

This subdirectory contains the source code regarding a projection of a function to an *hp*-mesh: the `Projection` class, which stores a function as nodal values on every spatial-angular *hp*-element of a `dg.mesh.Mesh`, its conversion to and from the global degree-of-freedom (DOF) vector and to and from files, integrals of a `Projection` in space or in angle, and the maps between the reference interval $[-1, 1]$ and an element.

Equation numbers below refer to the publication (Torchinsky, Du & Stechmann); numbers prefixed with `SI-` refer to its Supplemental Information.

How It Fits Into the Whole
--------------------------------------------------------------------------------

- `rt.Problem.solve` assembles and solves the linear system, merges the solution onto the `ROOT` MPI process with `dg.matrix.merge_vectors`, and returns it as a `Projection` built with `Projection(mesh)` followed by `Projection.from_vector`. On the other ranks it returns `None`.
- `amr.error_indicator.Error_Indicator` keeps the numerical solution in its `proj` attribute. The *hp*-steering criteria call `Projection.cell_intg_xy` (angular *hp*-steering, the function $g^K(\theta)$ of Section 3.4 and Eq. 18a) and `Projection.col_intg_th` (spatial *hp*-steering, Section SI-3). `Error_Indicator.to_file` calls `Projection.to_file`, and `amr.error_indicator.from_file` calls `projection.from_file`.
- `push_forward` and `pull_back` are used throughout `rt` (`get_Ex`, `get_Ey`, `get_Eth`, `boundary_conditions_vector`, and the matrix and vector assembly), in `amr.error_indicator.error` (`error_analytic`, `error_random`, `error_spatial_jump`), and in `Projection.__init__`.
- `tools.dg.projection` plots `Projection` objects.
- The experiment scripts in `experiments/` write each trial's solution with `Projection.to_file` (files `uh.npy` and `mesh.json`, and `uh_hr.npy` and `mesh_hr.json` for the high-resolution reference solution) and read them back with `projection.from_file`.

This subdirectory calls `dg.mesh` (`Mesh`, `mesh.from_file`, `remove_th`), `dg.quadrature` (`quad_xyth`, `lag_eval`) and `dg.matrix` (`get_idx_map`, `get_idx_inv_map`, `get_col_idxs`, `get_cell_idxs`). It does not use MPI: every function here works on a single process, and in the solver pipeline a `Projection` exists only on the `ROOT` process.

Data Layout
--------------------------------------------------------------------------------

A `Projection` mirrors the hierarchy of its `Mesh`:

- `Projection.cols` : `dict` mapping each column key to a `Projection_Column` (spatial element $C$).
- `Projection_Column.cells` : `dict` mapping each cell key to a `Projection_Cell` (angular element $K$ of that column).
- `Projection_Cell.vals` : `numpy` `ndarray` of shape `[nx, ny, nth]`, indexed `[i, j, a]`, where `[nx, ny] = Projection_Column.ndofs` and `[nth] = Projection_Cell.ndofs`.

`vals[i, j, a]` is the value of the function at the node $(x_i, y_j, \theta_a)$. These nodal values are the coefficients $u^{C,K}_{ija}$ of the Lagrange basis expansion of Eq. 6. The nodes are:

- $x_i$ (`i = 0, ..., nx - 1`) : the Legendre-Gauss-Lobatto (LGL) nodes `quad_xyth(nnodes_x = nx)[0]`, in increasing order on $[-1, 1]$ and therefore including both endpoints, mapped to $[x_0, x_1]$ with `push_forward`. `i = 0` is the left face and `i = nx - 1` the right face of the column.
- $y_j$ (`j = 0, ..., ny - 1`) : the LGL nodes `quad_xyth(nnodes_y = ny)[2]` mapped to $[y_0, y_1]$. `j = 0` is the bottom face and `j = ny - 1` the top face.
- $\theta_a$ (`a = 0, ..., nth - 1`) : the Legendre-Gauss (LG) nodes `quad_xyth(nnodes_th = nth)[4]`, in increasing order and not including the endpoints, mapped to $[\theta_0, \theta_1] \subseteq [0, 2\pi]$.

If the mesh has no angular dimension (`mesh.has_th == False`), each column has one cell with `ndofs = [1]`, `vals` has shape `[nx, ny, 1]`, and the function is evaluated at $\theta = 0$.

Global DOF Ordering
--------------------------------------------------------------------------------

This section and the next document conventions requested in issue #21.

`Projection.to_vector` flattens a `Projection` into one vector, and `Projection.from_vector` and the matrices and vectors built in `rt` use the same order:

1. Leaf columns in increasing column-key order (`dg.matrix.get_col_idxs`).
2. Within a column, leaf cells in increasing cell-key order (`dg.matrix.get_cell_idxs`).
3. Within a cell, the local index `beta = nth * ny * i + nth * j + a` (`dg.matrix.get_idx_map`): $\theta$ varies fastest and $x$ slowest. This is the C-order (row-major) flattening of `vals`, so `vals == cell_block.reshape([nx, ny, nth])`.

The global vector is the concatenation of the cell blocks in this order, and its length is `mesh.get_ndof()` $= \sum_C n_x^C n_y^C \sum_{K \in C} n_\theta^K$.

The local order differs from the one written in Section SI-9: Eqs. (22a, 22b) define $\alpha(i, j, a) = n_x n_y a + n_x j + i$ ($x$ fastest, $\theta$ slowest), whereas the code uses $\beta(i, j, a) = n_\theta n_y i + n_\theta j + a$. For example, with $n_x = n_y = 3$ and $n_\theta = 4$, the triple $(i, j, a) = (1, 0, 0)$ has index 12 in the code and 1 by Eq. (22a). The block order of Eqs. (23, 24) does not specify an order of columns or cells; the code uses increasing keys. Column keys and `Column.idx` are described in `dg/mesh/README.md`; note that `Column.idx` is stored as `[y index, x index]`.

On-Disk Format of Solution Files
--------------------------------------------------------------------------------

`Projection.to_file(file_path, write_mesh = True, mesh_file_path = "mesh.json")` writes two files:

- `mesh_file_path` (default `mesh.json`) : the mesh, written by `Mesh.to_file` as JSON. Column keys in `"cols"` and cell keys in each column's `"cells"` are JSON object keys, i.e. strings; sort them as integers (`sorted(keys, key = int)`) to recover the DOF order. Each column entry has `"ndofs": [nx, ny]` and `"pos": [x0, y0, x1, y1]`; each cell entry has `"ndofs": [nth]`, `"pos": [th0, th1]` and `"quadrant"`.
- `file_path` (default `projection.npy`; the experiments use `uh.npy` and `uh_hr.npy`) : the global vector of the previous section, written with `numpy.ndarray.tofile`. Despite the `.npy` extension this is **not** an NPY file: it is a raw dump of `8 * ndof` bytes of `float64` values in the byte order of the machine that wrote it, with no header, dtype or shape. `numpy.load` raises a `ValueError`, because without the NPY header it treats the file as pickled data. Read it with `numpy.fromfile(path, dtype = "<f8")` (little-endian `float64`, the byte order of x86-64 and AArch64 machines), together with the mesh file.

`projection.from_file` reads the pair back with `numpy.fromfile(path)` (default dtype `float64`, native byte order).

Reading a solution with this package:

```python
from dg.projection import from_file

uh = from_file("out/trial/mesh.json", "out/trial/uh.npy")
vals = uh.cols[col_key].cells[cell_key].vals # Shape [nx, ny, nth]
```

Reading a solution without this package:

```python
import json
import numpy as np

mesh = json.load(open("mesh.json"))
uh = np.fromfile("uh.npy", dtype = "<f8")

g_idx = 0
blocks = {} # (col_key, cell_key) -> array of shape [nx, ny, nth]
for col_key in sorted(mesh["cols"], key = int):
    col = mesh["cols"][col_key]
    [nx, ny] = col["ndofs"]
    for cell_key in sorted(col["cells"], key = int):
        [nth] = col["cells"][cell_key]["ndofs"]
        n = nx * ny * nth
        blocks[(int(col_key), int(cell_key))] = uh[g_idx:g_idx + n].reshape([nx, ny, nth])
        g_idx += n
assert g_idx == uh.size
```

(All columns and cells stored by `Projection.to_file` are leaves, since `Projection` requires a mesh of leaves.)

Source Files
--------------------------------------------------------------------------------

The source files contained in the `projection` subdirectory are:

- `__init__.py` : Adds the `src` directory to `sys.path` and exports the subpackage `projection_column` and the names `Projection`, `from_file`, `get_f2f_matrix`, `push_forward` and `pull_back`. `intg_th` is not exported; import it with `from dg.projection.intg_th import intg_th`.
- `class_Projection` : Class implementing the projection of a function to an *hp*-mesh.
- `cell_intg_xy` : Method `Projection.cell_intg_xy`, which integrates an angular-spatial *hp*-element (`cell`) in space.
- `col_intg_th` : Method `Projection.col_intg_th`, which integrates the numerical solution on the angular *hp*-mesh of a spatial *hp*-element (`column`, or `col`) in angle.
- `from_file` : Reads a `Projection` from a file.
- `from_vector` : Method `Projection.from_vector`, which fills a `Projection` from a `numpy` `ndarray`.
- `get_f2f_matrix` : Returns a matrix for mapping between faces of neighboring spatial-angular *hp*-elements.
- `intg_th` : Integrates a `Projection` in angle, returning a `Projection` on a mesh without the angular dimension.
- `push_pull` : Maps between the intervals $\left[ x_0,\ x_f \right]$ and $\left[ -1,\ 1 \right]$.
- `to_file` : Method `Projection.to_file`, which writes a `Projection` to a file.
- `to_vector` : Method `Projection.to_vector`, which converts a `Projection` to a `numpy` `ndarray`.

Functions and Classes
--------------------------------------------------------------------------------

### `class Projection(mesh, func = None)` (`class_Projection.py`)

Evaluates a function at the nodes of every spatial-angular *hp*-element of a mesh (see Data Layout). This is nodal interpolation: `Projection` evaluates `func` at the nodes, and does not compute an $L^2$ projection.

- `mesh` (`dg.mesh.Mesh`) : The mesh. It is deep-copied into `self.mesh`, so later changes to `mesh` do not affect the `Projection`. Every column and every cell must be a leaf (`is_lf == True`); otherwise an `AssertionError` is raised.
- `func` (callable or `None`, default `None`) : The function to evaluate. If `mesh.has_th` is `True` it is called as `func(x, y, th)`, otherwise as `func(x, y)`. It is called once per node with scalar `float` arguments and must return a scalar. If `None`, all values are zero.

Attributes:

- `mesh` (`Mesh`) : Deep copy of the input mesh.
- `cols` (`dict`) : Column key (`int`) to `Projection_Column`.

Methods:

- `cell_intg_xy(col_key, cell_key)`, `col_intg_th(col_key)`, `to_vector()`, `from_vector(vec)`, `to_file(file_path, **kwargs)` : Documented below.
- `__eq__(other)` : `True` if `self.mesh == other.mesh` and `self.cols == other.cols` (exact equality of all `vals`). Comparing projections whose cells have different `ndofs` raises a `ValueError` from `numpy` instead of returning `False` (see `Projection_Cell.__eq__`).
- `__str__()` : Lists `Ls`, `pbcs` and `has_th` of the mesh and the sorted column keys.

Example:

```python
import numpy as np
from dg.mesh import Mesh
from dg.projection import Projection

mesh = Mesh(Ls = [3., 2.], pbcs = [False, False], ndofs = [3, 3, 4], has_th = True)
mesh.ref_mesh(kind = "ang", form = "h")
proj = Projection(mesh, lambda x, y, th: np.exp(-x) * np.cos(th)**2)
proj.cols[0].cells[1].vals.shape # (3, 3, 4): column 0, first of the two angular cells
```

### `Projection.cell_intg_xy(col_key, cell_key) -> np.ndarray` (`cell_intg_xy.py`)

Integrates one cell in space at each of its angular nodes:

$$\mathtt{out}[a] = \frac{\Delta x \Delta y}{4} \sum_{i=0}^{n_x-1} \sum_{j=0}^{n_y-1} w_i^x w_j^y\, \mathtt{vals}[i, j, a] \approx \int_{y_0}^{y_1} \int_{x_0}^{x_1} u(x, y, \theta_a)\, dx\, dy,$$

with the LGL weights of `quad_xyth` (Eq. SI-8 (15)).

- `col_key` (`int`) : Key of a leaf column.
- `cell_key` (`int`) : Key of a leaf cell in that column.
- Returns : `ndarray` of shape `[nth]`.

Used by `amr.error_indicator.cell_hp_steer` for the angular *hp*-steering criterion.

### `Projection.col_intg_th(col_key) -> np.ndarray` (`col_intg_th.py`)

Integrates a column in angle over all of its cells at each spatial node:

$$\mathtt{out}[i, j] = \sum_{K \in C} \frac{\Delta \theta^K}{2} \sum_{a=0}^{n_\theta^K - 1} w_a^K\, \mathtt{vals}^K[i, j, a] \approx \int_0^{2\pi} u(x_i, y_j, \theta)\, d\theta,$$

with the LG weights of `quad_xyth`.

- `col_key` (`int`) : Key of a leaf column.
- Returns : `ndarray` of shape `[nx, ny]`.

Used by `amr.error_indicator.col_hp_steer` for the spatial *hp*-steering criterion and by `intg_th`.

### `Projection.to_vector() -> np.ndarray` (`to_vector.py`)

Returns the global DOF vector (see Global DOF Ordering): a 1-D `float64` `ndarray` of length `self.mesh.get_ndof()`. It passes the `Projection` itself to `get_col_idxs` and `get_cell_idxs`, which only need its `cols` and `is_lf` attributes.

### `Projection.from_vector(vec) -> Projection` (`from_vector.py`)

Overwrites the values of the `Projection` with a global DOF vector, in place.

- `vec` (`ndarray`, 1-D) : Global DOF vector in the order of Global DOF Ordering. Entries past `mesh.get_ndof()` are ignored; a shorter vector raises an `IndexError`.
- Returns : `self` (the annotation says `None`). Each `Projection_Cell.vals` is replaced by a new array of shape `[nx, ny, nth]`.

### `Projection.to_file(file_path = "projection.npy", **kwargs) -> None` (`to_file.py`)

Writes the projection and, optionally, its mesh. See On-Disk Format of Solution Files.

- `file_path` (`str`, default `"projection.npy"`) : Output path of the raw `float64` vector.
- `write_mesh` (`bool`, keyword, default `True`) : Also write the mesh with `Mesh.to_file`.
- `mesh_file_path` (`str`, keyword, default `"mesh.json"`) : Output path of the mesh.
- Other keyword arguments are ignored (`Error_Indicator.to_file` passes its own keyword arguments through).

Side effects: writes files, overwriting existing ones. There is no MPI communication; in the solver pipeline it is called on the `ROOT` process, which holds the `Projection`.

### `from_file(mesh_file_path = "mesh.json", projection_file_path = "projection.npy") -> Projection` (`from_file.py`)

Reads a mesh with `dg.mesh.from_file`, reads the vector with `numpy.fromfile` (`float64`, native byte order), and returns `Projection(mesh).from_vector(vector)`.

- `mesh_file_path` (`str`) : Path of the mesh JSON file.
- `projection_file_path` (`str`) : Path of the raw `float64` vector written by `Projection.to_file`.

### `intg_th(projection) -> Projection` (`intg_th.py`)

Integrates a `Projection` in angle. It builds the mesh without the angular dimension with `dg.mesh.remove_th` and returns a `Projection` on it whose single cell per column holds `col_intg_th(col_key)` reshaped to `[nx, ny, 1]`.

- `projection` (`Projection`) : A projection on a mesh with `has_th == True`.

Not exported by `__init__.py` and not used by the package; `tests/test_dg/test_helpers.py` tests it.

### `get_f2f_matrix(dim_str, nbasis, nnode, nhbr_rel) -> np.ndarray` (`get_f2f_matrix.py`)

Returns the face-to-face matrix $F$ of shape `[nbasis, nnode]` with entries $F_{mn} = c\, \zeta^0_m(z_n)$: the $m$-th Lagrange basis function of element `_0` (with `nbasis` nodes) evaluated at the $n$-th node of the neighboring element `_1` (with `nnode` nodes), where $z_n$ is that node expressed in the reference coordinate of `_0`. Entries with magnitude below $10^{-14}$ are set to zero.

- `dim_str` (`str`) : `"x"` or `"y"` (LGL nodes) or `"th"` (LG nodes). Any other value raises an `UnboundLocalError`.
- `nbasis` (`int`) : Number of nodes (basis functions) of `_0`.
- `nnode` (`int`) : Number of nodes of `_1`.
- `nhbr_rel` (`tuple`) : `(level relation, position)`. It must be a `tuple`, not a `list` (it is used as a dictionary key; the docstring calls it a list, which raises `TypeError: unhashable type`). Allowed values, with $[z_0, z_1]$ the interval of `_1` in the reference coordinate of `_0`:
  - `(0, "s")` : Same refinement level; $[z_0, z_1] = [-1, 1]$. If also `nbasis == nnode`, returns the identity matrix.
  - `(1, "u")` / `(1, "l")` : `_0` is less refined and `_1` is its upper / lower half; $[z_0, z_1] = [0, 1]$ / $[-1, 0]$; $c = 1$.
  - `(-1, "u")` / `(-1, "l")` : `_0` is more refined and is the upper / lower half of `_1`; $[z_0, z_1] = [-3, 1]$ / $[-1, 3]$. The nodes of `_1` are first compressed to the half of its reference interval that overlaps `_0` ($[0, 1]$ / $[-1, 0]$), so the points $z_n$ cover $[-1, 1]$; $c = 1/2$, the Jacobian of the half interval.
  - Any other value prints an error and calls `sys.exit(1)`.

Side effects: results are cached in the module-level `dict` `f2f_matrices`, keyed by `(dim_str, nbasis, nnode, nhbr_rel)`, and the cached array itself is returned, so modifying the result changes later calls. `rt/get_Ey.py` imports `get_f2f_matrix`, but no code in `src` or `tests` calls it; `rt.get_Ex` and `rt.get_Ey` build the equivalent matrices themselves.

### `push_forward(x0, xf, nodes) -> np.ndarray` (`push_pull.py`)

Maps points from the reference interval $[-1, 1]$ to $[x_0, x_f]$: $x = x_0 + \frac{x_f - x_0}{2} (\bar{x} + 1)$.

- `x0`, `xf` (`float`) : Endpoints of the target interval.
- `nodes` (`float` or `ndarray`) : Points in $[-1, 1]$.
- Returns : Points in $[x_0, x_f]$, same shape as `nodes`.

This is the push-forward map of Eq. SI-8 (12). As printed in the Supplemental Information, Eq. SI-8 (12) reads $\frac{\Delta z^E}{2}(\bar{z} + 1) - z_0^E$; the code adds $x_0$.

### `pull_back(x0, xf, nodes) -> np.ndarray` (`push_pull.py`)

Maps points from $[x_0, x_f]$ to $[-1, 1]$: $\bar{x} = -1 + \frac{2}{x_f - x_0} (x - x_0)$, the pull-back map of Eq. SI-8 (13). Arguments as for `push_forward`; points outside $[x_0, x_f]$ map outside $[-1, 1]$.

```python
import numpy as np
from dg.projection import push_forward, pull_back

xx = push_forward(1., 3., np.array([-1., 0., 1.])) # [1., 2., 3.]
pull_back(1., 3., xx)                               # [-1., 0., 1.]
```

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `projection` subdirectory is

```
projection
|
|--- projection_column : Contains source code implementing the Projection_Column class (spatial elements of a Projection).
     |
     |--- projection_cell : Contains source code implementing the Projection_Cell class (angular elements of a Projection).
```
