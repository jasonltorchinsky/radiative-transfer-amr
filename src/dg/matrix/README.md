`matrix`
================================================================================

This subdirectory contains the source code to index degrees of freedom (DOFs) in global vectors and matrices, and to handle matrices and vectors across multiple MPI processes: masks that separate the inflow-boundary DOFs from the other DOFs, and the splitting and merging of `petsc4py` matrices and vectors with those masks.

Equation numbers below refer to the publication (Torchinsky, Du & Stechmann); numbers prefixed with `SI-` refer to its Supplemental Information. The index conventions and masks are documented here per issue #21.

How It Fits Into the Whole
--------------------------------------------------------------------------------

- The index maps (`get_idx_map`, `get_col_idxs`, `get_cell_idxs`) are used by every assembly routine in `rt` (`mass_matrix`, `scattering_matrix`, `preconditioner_matrix`, `interior_convection_matrix`, `boundary_convection_matrix`, `forcing_vector`) and by `dg.projection` (`Projection.to_vector`; `Projection.from_vector` uses `get_idx_inv_map`). They fix the order of the global DOF vector and of the rows and columns of the global matrix.
- `rt.Problem.solve` uses the rest of this subdirectory to impose the inflow boundary condition:
  1. `get_intr_mask(mesh)` marks the DOFs on the inflow part of the domain boundary.
  2. `split_matrix(sys_mat, intr_mask)` returns the interior-interior block $A_{II}$ and the interior-boundary block $A_{IB}$ of the system matrix.
  3. `split_vector` returns the interior part $f_I$ of the forcing vector and the boundary part $g_B$ of the boundary-conditions vector (nodal values of the inflow data).
  4. PETSc solves $A_{II} u_I = f_I - A_{IB} g_B$.
  5. `merge_vectors(u_I, g_B, intr_mask)` assembles the full solution vector on the root process, which `solve` converts to a `dg.projection.Projection`.

  The rows of the boundary DOFs are discarded, and the boundary DOFs take the nodal values of the inflow data. The publication describes the inflow data entering through the upwind numerical trace $\hat{u}$ on the domain boundary (Eqs. 4 and 12); the code imposes the inflow values at the boundary nodes as described here.

This subdirectory calls `dg.mesh` (for `Mesh` and `Column`), `consts` (`COMM_ROOT`, `INT`), `utils.print_msg`, `numpy`, `mpi4py` and `petsc4py`. All MPI communication uses `MPI.COMM_WORLD` / `PETSc.COMM_WORLD`.

Global DOF Ordering
--------------------------------------------------------------------------------

The global DOF vector (and each dimension of the global matrix) is ordered as follows:

1. Leaf columns in increasing column-key order; `get_col_idxs` gives each column's position.
2. Within a column, leaf cells in increasing cell-key order; `get_cell_idxs` gives each cell's position.
3. Within the spatial-angular element $(C, K)$ with `[nx, ny] = col.ndofs` and `[nth] = cell.ndofs`, the nodal value $u^{C,K}_{ija}$ (Eq. 6; `i` indexes the $x$ nodes, `j` the $y$ nodes, `a` the $\theta$ nodes, each in increasing order) has local index

   `beta(i, j, a) = nth * ny * i + nth * j + a`,

   so $\theta$ varies fastest and $x$ slowest. This equals the C-order (row-major) flattening of an array of shape `[nx, ny, nth]`, e.g. `Projection_Cell.vals.reshape(-1)`.

The vector is the concatenation of the element blocks, each of length `nx * ny * nth`, in this order; its total length is `mesh.get_ndof()`.

The local order differs from Section SI-9, where Eqs. (22a, 22b) define $\alpha(i, j, a) = n_x n_y a + n_x j + i$ and $\beta(p, q, r) = n_x n_y r + n_x q + p$ ($x$ fastest, $\theta$ slowest). For $n_x = n_y = 3$, $n_\theta = 4$:

| $(i, j, a)$ | `get_idx_map` | Eq. (22a) |
| --- | --- | --- |
| $(0, 0, 1)$ | 1 | 9 |
| $(0, 1, 0)$ | 4 | 3 |
| $(1, 0, 0)$ | 12 | 1 |

Eqs. (23, 24) assemble element matrices into column and global block matrices without specifying the order of cells and columns; the code uses increasing keys. The docstrings of `get_idx_map` and `get_idx_inv_map` name the maps "p,q,r => alpha, i,j,a => beta", the reverse of the naming in Eqs. (22a, 22b); both triples use the same map.

Interior and Boundary Masks
--------------------------------------------------------------------------------

The interior mask is a 1-D `bool` array of length `mesh.get_ndof()` in the global DOF order. An entry is `False` (a boundary DOF) when the DOF is a node on the inflow part of the spatial domain boundary, and `True` (an interior DOF) otherwise. For each leaf column and each leaf cell with angular quadrant $q$ (`cell.quadrant`, with $[\theta_0, \theta_1] \subseteq [q\pi/2, (q+1)\pi/2]$), a face of the column is on the inflow boundary when the column has no neighbor through it (`col.nhbr_keys[F] == [None, None]`, i.e. a non-periodic domain boundary) and the directions of the quadrant point into the domain through it:

| Face `F` | Face | Inflow quadrants $q$ | Masked nodes (all `a`) |
| --- | --- | --- | --- |
| 0 | right, $x = x_1$ | 1, 2 ($\cos\theta \le 0$) | `i = nx - 1`, all `j` |
| 1 | top, $y = y_1$ | 2, 3 ($\sin\theta \le 0$) | `j = ny - 1`, all `i` |
| 2 | left, $x = x_0$ | 3, 0 ($\cos\theta \ge 0$) | `i = 0`, all `j` |
| 3 | bottom, $y = y_0$ | 0, 1 ($\sin\theta \ge 0$) | `j = 0`, all `i` |

These are the DOFs with $\vec{s} \cdot \hat{n}_{\partial\Omega} \le 0$ of Eq. 4. Because the $x$ and $y$ nodes are Legendre-Gauss-Lobatto nodes, `i = 0` and `i = nx - 1` (and likewise for `j`) lie on the faces; this requires `nx, ny >= 2`. Cells with `quadrant == None` (spanning more than one quadrant) are never masked. `rt.Problem.solve` rejects meshes that violate either condition (`rt.solve.check_mesh`). A corner node of an inflow corner is masked by both faces. The boundary mask is the elementwise negation of the interior mask.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `matrix` subdirectory are:

- `__init__.py` : Adds the `src` directory to `sys.path` and exports `get_idx_map`, `get_idx_inv_map`, `get_col_idxs`, `get_cell_idxs`, `get_intr_mask`, `get_bdry_mask`, `merge_vectors`, `split_matrix` and `split_vector`.
- `get_idxs` : Contains indexing maps for, e.g., unrolling tensors into vectors, constructing global vectors/matrices, etc.
- `get_masks` : Contains functions for obtaining masks for vectors/matrices to obtain only the entries corresponding to interior/boundary values.
- `merge_vectors` : Merge interior and boundary vectors split across multiple MPI processes into a single vector on the root process.
- `split_matrix` : Split a matrix distributed across the MPI processes into its interior-interior and interior-boundary blocks.
- `split_vector` : Split a vector distributed across the MPI processes into its interior and boundary parts.

Functions
--------------------------------------------------------------------------------

### `get_idx_map(ndof_x, ndof_y, ndof_th)` (`get_idxs.py`)

Returns the function `idx_map(ii, jj, aa) = ndof_th * ndof_y * ii + ndof_th * jj + aa`, the local index of node `(ii, jj, aa)` within an element (see Global DOF Ordering).

- `ndof_x`, `ndof_y`, `ndof_th` (`int`) : Number of nodes in $x$, $y$, $\theta$ of the element. `ndof_x` does not enter the map; it is kept in case the map changes.
- Returns : A function of three `int`s returning an `int` in `[0, ndof_x * ndof_y * ndof_th)`.

```python
from dg.matrix import get_idx_map, get_idx_inv_map

beta = get_idx_map(3, 3, 4)
beta(1, 0, 0)                      # 12
get_idx_inv_map(3, 3, 4)(12)       # [1, 0, 0]
```

### `get_idx_inv_map(ndof_x, ndof_y, ndof_th)` (`get_idxs.py`)

Returns the inverse of `get_idx_map`: the function `idx_inv_map(idx)`, which returns `[i, j, a]` (list of `int`) with `a = idx % ndof_th`, `j = (idx // ndof_th) % ndof_y` and `i = idx // (ndof_th * ndof_y)`. Arguments as for `get_idx_map`; `ndof_x` is unused.

### `get_col_idxs(mesh) -> list` (`get_idxs.py`)

Numbers the leaf columns in increasing key order.

- `mesh` (`Mesh`, or any object with a `cols` dictionary of objects with an `is_lf` attribute; `Projection.to_vector` passes a `Projection`).
- Returns : `[ncols, col_idxs]`, where `ncols` (`int`) is the number of leaf columns and `col_idxs` (`dict`) maps each leaf column key to its position `0, ..., ncols - 1`.

### `get_cell_idxs(mesh, col_key) -> list` (`get_idxs.py`)

Numbers the leaf cells of one column in increasing key order.

- `mesh` (`Mesh` or `Projection`, as for `get_col_idxs`).
- `col_key` (`int`) : Key of the column.
- Returns : `[ncells, cell_idxs]`, where `ncells` (`int`) is the number of leaf cells and `cell_idxs` (`dict`) maps each leaf cell key to its position `0, ..., ncells - 1`.

### `get_intr_mask(mesh, **kwargs) -> np.ndarray` (`get_masks.py`)

Returns `get_intr_mask_seq(mesh, **kwargs)`. This is the mask used by `rt.Problem.solve`.

### `get_intr_mask_seq(mesh, verbose = False, blocking = True) -> np.ndarray` (`get_masks.py`)

Builds the interior mask (see Interior and Boundary Masks) on the root process only.

- `mesh` (`Mesh`) : The mesh; it is read only on the root process (`consts.COMM_ROOT`).
- `verbose` (`bool`, keyword, default `False`) : Print progress and timing with `utils.print_msg`.
- `blocking` (`bool`, keyword, default `True`) : Call `MPI.COMM_WORLD.Barrier()` before returning.
- Other keyword arguments are ignored.
- Returns : On the root process, the `bool` `ndarray` of length `mesh.get_ndof()`. On the other processes, the `int` `0`.

Side effects: initializes `petsc4py` on `MPI.COMM_WORLD` if it is not initialized. With `blocking = True` (the default) it is collective: every process must call it.

### `get_intr_mask_mpi(mesh, verbose = False, blocking = True) -> np.ndarray` (`get_masks.py`)

Builds the same mask in parallel. The mesh is broadcast from the root process, the leaf column keys are split across the processes with `numpy.array_split`, each process builds the masks of its columns, and the column masks are combined with `MPI.COMM_WORLD.allgather`.

- Arguments as for `get_intr_mask_seq`; `mesh` need only be valid on the root process.
- Returns : The full mask on every process.

Side effects: calls `petsc4py.init()`; collective (`bcast`, `allgather`, and a barrier if `blocking`). Not exported by `__init__.py` and not used by the package; `tests/test_dg/test_helpers.py` checks that it equals `get_intr_mask_seq`.

### `get_bdry_mask(mesh) -> np.ndarray` (`get_masks.py`)

Returns `numpy.invert(get_intr_mask(mesh))`, the boundary mask, on the root process. It calls `get_intr_mask` with default keyword arguments, so it is collective. On the other processes `get_intr_mask` returns `0` and `get_bdry_mask` returns `numpy.invert(0) == -1`, not a mask. Not used by the package.

### `split_matrix(matrix, intr_mask) -> list` (`split_matrix.py`)

Splits the global system matrix into its interior-interior and interior-boundary blocks.

- `matrix` (`PETSc.Mat`, square, global size `ndof`, rows distributed over `PETSc.COMM_WORLD`) : The system matrix in the global DOF order.
- `intr_mask` (`ndarray` of `bool`, length `ndof`) : The interior mask. Only the value on the root process is used; it is broadcast to the others (they may pass the `0` returned by `get_intr_mask`).
- Returns : `[matrix_intr, matrix_bdry]`, both `PETSc.Mat`: `matrix_intr` has the interior rows and interior columns ($A_{II}$), `matrix_bdry` the interior rows and boundary columns ($A_{IB}$). Rows and columns keep the global order within each set. Each process owns the interior rows that lay in its ownership range of `matrix`.

Side effects: collective (`bcast` of the interior and boundary global indices, `createSubMatrix`). The `PETSc.IS` index sets are not destroyed explicitly.

### `split_vector(vector, intr_mask) -> list` (`split_vector.py`)

Splits a global vector into its interior and boundary parts.

- `vector` (`PETSc.Vec`, global size `ndof`, distributed over `PETSc.COMM_WORLD`) : A vector in the global DOF order.
- `intr_mask` : As for `split_matrix`.
- Returns : `[vector_intr, vector_bdry]`, both `PETSc.Vec`, obtained with `vector.getSubVector`. Each process holds the entries of its ownership range of `vector`.

Side effects: collective (`bcast`, `getSubVector`). The sub-vectors are not returned to `vector` with `restoreSubVector`.

### `merge_vectors(vector_intr, vector_bdry, intr_mask) -> np.ndarray` (`merge_vectors.py`)

Inverse of `split_vector`: gathers an interior vector and a boundary vector onto process 0 and interleaves them according to the mask.

- `vector_intr` (`PETSc.Vec`) : Interior values (in `rt.Problem.solve`, the solution $u_I$), length `intr_mask.sum()`.
- `vector_bdry` (`PETSc.Vec`) : Boundary values (the inflow data $g_B$), length `(~intr_mask).sum()`.
- `intr_mask` (`ndarray` of `bool`, length `ndof`) : Used only on process 0.
- Returns : On process 0, a `float64` `ndarray` of length `ndof` in the global DOF order, with the interior entries where `intr_mask` is `True` and the boundary entries elsewhere. On the other processes, an empty array.

Side effects: collective (two `MPI.COMM_WORLD.gather` calls). The root is hard-coded as process 0, whereas `split_matrix` and `split_vector` use `consts.COMM_ROOT`; both are `0`.

Example (on every MPI process, after building the PETSc system matrix `A`, forcing vector `f` and boundary-conditions vector `g`):

```python
from dg.matrix import get_intr_mask, split_matrix, split_vector, merge_vectors

intr_mask = get_intr_mask(mesh)                 # bool array on root, 0 elsewhere
[A_II, A_IB] = split_matrix(A, intr_mask)
[f_I, _] = split_vector(f, intr_mask)
[_, g_B] = split_vector(g, intr_mask)
# ... solve A_II u_I = f_I - A_IB g_B with PETSc ...
uh_vec = merge_vectors(u_I, g_B, intr_mask)     # full vector on process 0
```
