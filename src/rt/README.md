`rt`
================================================================================

This subdirectory contains the source code regarding numerically solving the steady-state monochromatic radiative transfer equation (RTE). It defines the `Problem` class, which holds the coefficients, phase function, source and inflow data of an RTE problem, and the methods of `Problem` that assemble the discontinuous Galerkin (DG) spectral element system on a `dg.mesh.Mesh` and solve it with PETSc.

Equation numbers below refer to the publication (Torchinsky, Du & Stechmann); numbers prefixed with `SI-` refer to its Supplemental Information. The boundary-condition part of documentation issue #20 is addressed in the section "Inflow Boundary Conditions".

How It Fits Into the Whole
--------------------------------------------------------------------------------

The package solves the 2-D-space, 1-D-angle RTE (Eq. 3)

$$\vec{s} \cdot \nabla_{\vec{x}} u(\vec{x}, \theta) + \kappa(\vec{x})\, u(\vec{x}, \theta) - \sigma(\vec{x}) \int_0^{2\pi} \Phi(\theta, \theta')\, u(\vec{x}, \theta')\, d\theta' = f(\vec{x}, \theta),$$

with $\vec{s} = (\cos\theta, \sin\theta)$, on $[0, L_x] \times [0, L_y] \times [0, 2\pi]$, with inflow data $u = u^{in,bd}$ where $\vec{s} \cdot \hat{n}_{\partial\Omega} \le 0$ (Eq. 4).

- Callers: the experiment scripts (`experiments/*/problem.py` build a `Problem`; `experiments/*/experiment.py` call `problem.solve(mesh, **solver_params)` once per refinement step), `amr.error_indicator.error.error_high_resolution` (solves the same `Problem` on a uniformly refined mesh as a reference solution), and the tests in `tests/test_rt`.
- In the adaptive loop, `Problem.solve` produces the numerical solution $u_{hp}$ as a `dg.projection.Projection`. The `amr` package computes error indicators from it and refines the `Mesh`, and the loop calls `solve` again on the new mesh. Nothing in `rt` keeps state between solves apart from the module-level caches of `get_Ex`, `get_Ey` and `interior_convection_matrix`, which depend only on node counts and relative element positions.
- `rt` calls `dg.mesh` (`Mesh`, `Column`, `Cell`, `Mesh.get_ndof`, `Mesh.nhbr_cells_in_nhbr_col`), `dg.quadrature` (`quad_xyth`, `lag_eval`, `lag_ddx_eval`), `dg.projection` (`push_forward`, `pull_back`, `Projection`), `dg.matrix` (DOF index maps, the inflow-boundary mask, and the matrix and vector splitting/merging), `consts` (`EPS`, `INF`, `INT`, `PI`, `COMM_ROOT`) and `utils.print_msg`, together with `numpy`, `scipy.sparse`, `mpi4py` and `petsc4py`.

Conventions
--------------------------------------------------------------------------------

- **Coordinates.** A column (spatial element) $C$ has `col.pos = [x0, y0, x1, y1]`; a cell (angular element) $K$ has `cell.pos = [th0, th1]` with $0 \le \theta_0 < \theta_1 \le 2\pi$. The nodes are Legendre-Gauss-Lobatto (LGL) nodes in $x$ and $y$ (so `i = 0` and `i = nx - 1` lie on the left and right faces) and Legendre-Gauss (LG) nodes in $\theta$, all in increasing order (`dg.quadrature.quad_xyth`). Physical nodes are `push_forward(x0, x1, xxb)`; reference weights `w` are on $[-1, 1]$, so each 1-D integral carries a factor $\Delta/2$.
- **DOF order.** Global vectors and the rows and columns of global matrices follow `dg.matrix`: leaf columns by increasing key, leaf cells by increasing key within a column, then node $(i, j, a)$ of the element at local index `beta(i, j, a) = nth * ny * i + nth * j + a` ($\theta$ fastest). This differs from Eqs. (22a, 22b) of SI-9 ($x$ fastest); see `src/dg/matrix/README.md`.
- **Faces.** `F = 0, 1, 2, 3` are the right ($x = x_1$), top ($y = y_1$), left ($x = x_0$) and bottom ($y = y_0$) faces of a column, with outward normal $(\cos(F\pi/2), \sin(F\pi/2))$, so $\vec{s} \cdot \hat{n}_F = \Theta_F(\theta) = \cos(\theta - F\pi/2)$ (Eq. SI-41). `col.nhbr_keys[F]` holds up to two neighboring column keys through face `F` (`[None, None]` on a non-periodic domain boundary; the column itself for a periodic single-column direction).
- **Quadrants.** `cell.quadrant = q` means $[\theta_0, \theta_1] \subseteq [q\pi/2, (q+1)\pi/2]$ (Eq. SI-39); it is `None` for cells spanning more than one quadrant. Face `F` is an outflow face ($\Theta_F \ge 0$ on the whole cell) for `q = F` and `q = F - 1 (mod 4)`, and an inflow face otherwise:

  | `q` | Outflow faces | Inflow faces |
  | --- | --- | --- |
  | 0 ($[0, \pi/2]$) | 0, 1 | 2, 3 |
  | 1 ($[\pi/2, \pi]$) | 1, 2 | 3, 0 |
  | 2 ($[\pi, 3\pi/2]$) | 2, 3 | 0, 1 |
  | 3 ($[3\pi/2, 2\pi]$) | 3, 0 | 1, 2 |

  This is Table 1 of SI-13. Every leaf cell must have a quadrant, which `solve` checks.
- **MPI.** All communication uses `MPI.COMM_WORLD` / `PETSc.COMM_WORLD`, and the root is `consts.COMM_ROOT` (0). Every rank must call the assembly methods and `solve` with the same `Mesh`. Unless stated otherwise, an assembly method evaluates `mesh.get_ndof()` on the root and broadcasts it, gives each rank the columns `np.array_split(sorted(mesh.cols.keys()), comm_size)[comm_rank]`, inserts that rank's entries into a distributed PETSc object with global indices (PETSc moves off-process entries during `assemblyBegin`/`assemblyEnd`), and ends with `MPI_comm.Barrier()` when `blocking = True`.
- **Return types.** Some type hints in the source name `scipy.sparse` matrices or `np.ndarray`; the matrix methods return a distributed `PETSc.Mat` (AIJ, size `[ndof, ndof]`) and the vector methods a distributed `PETSc.Vec` (size `ndof`).

The Discrete System
--------------------------------------------------------------------------------

`solve` assembles the matrix of the bilinear form $a(u, v)$ of Eq. (11a) as

`sys_mat = (bc_mat - ic_mat) + ma_sc_mat`,

that is, boundary convection minus interior convection plus extinction minus scattering, and the right-hand side from the forcing term (Eq. 11b). For an element $(C, K)$ with widths $\Delta x, \Delta y, \Delta\theta$, reference weights $w^x_i, w^y_j, w^\theta_a$ and nodes $(x_i, y_j, \theta_a)$, the entries (row = test node, column = trial node) are:

| Term | Method | Entry | Paper |
| --- | --- | --- | --- |
| Extinction $M_\kappa$ | `mass_matrix` | $[\beta(i,j,a), \beta(i,j,a)] = \frac{\Delta x \Delta y \Delta\theta}{8} w^x_i w^y_j w^\theta_a\, \kappa(x_i, y_j)$ | Eq. SI-28 |
| Scattering $-M_\sigma$ | `scattering_matrix` | $[\alpha(i,j,r), \beta(i,j,a)] = -\frac{\Delta x \Delta y \Delta\theta^K \Delta\theta^{K'}}{16} w^x_i w^y_j w^{\theta,K}_r w^{\theta,K'}_a\, \sigma(x_i, y_j)\, \Phi(\theta^K_r, \theta^{K'}_a)$ for every pair of cells $K, K'$ of the same column | Eq. SI-34 |
| $M_\kappa - M_\sigma$ | `preconditioner_matrix` | sum of the two rows above | |
| Interior convection $I$ | `interior_convection_matrix` | $[\alpha(p,j,a), \beta(i,j,a)] \mathrel{+}= \frac{\Delta y \Delta\theta}{4} w^x_i w^y_j w^\theta_a \cos\theta_a\, \bar\phi'_p(\bar x_i)$ and $[\alpha(i,q,a), \beta(i,j,a)] \mathrel{+}= \frac{\Delta x \Delta\theta}{4} w^x_i w^y_j w^\theta_a \sin\theta_a\, \bar\psi'_q(\bar y_j)$ | Eq. SI-37 |
| Boundary convection $B$ | `boundary_convection_matrix` | face terms, see below | Eqs. SI-40 to SI-52 |
| Forcing $F$ | `forcing_vector` | $[\beta(i,j,a)] = \frac{\Delta x \Delta y \Delta\theta}{8} w^x_i w^y_j w^\theta_a\, f(x_i, y_j, \theta_a)$ | Eq. 11b |
| Inflow data $g$ | `boundary_conditions_vector` | $[\beta(i,j,a)] = u^{in,bd}(x_i, y_j, \theta_a)$ (nodal values) | Eq. 4 |

Here $\bar\phi'_p(\bar x_i)$ is the derivative of the $p$-th reference Lagrange basis function at the $i$-th reference node; the factor $2/\Delta x$ of the physical derivative is folded into the coefficient. The scattering matrix couples all cells of a column and no cells of different columns, so $M_\kappa - M_\sigma$ is block diagonal by column (dense within a column, up to the threshold below). The interior convection and extinction matrices are block diagonal by element.

Boundary convection on face `F` of element $(C, K)$ ($\Delta_F = \Delta y$ for `F = 0, 2` and $\Delta x$ for `F = 1, 3`; "face nodes" are `i = nx - 1` for `F = 0`, `j = ny - 1` for `F = 1`, `i = 0` for `F = 2`, `j = 0` for `F = 3`):

- **Outflow face** (by the quadrant table): own-trace term on the face nodes, e.g. for `F = 0`, $[\beta(n_x - 1, j, a), \beta(n_x - 1, j, a)] \mathrel{+}= \frac{\Delta_F \Delta\theta}{4} w^y_j w^\theta_a \Theta_F(\theta_a)$ (Eqs. SI-47a, SI-48a, SI-50a). This is assembled whether or not the face lies on $\partial\Omega$.
- **Inflow face with a neighboring column $C'$** (interior face, or a periodic face): upwind term coupling the face nodes of $(C, K)$ (rows) to the opposite face nodes of each neighboring cell $K'$ of $C'$ (columns), e.g. for `F = 0`, $[\alpha(n_x - 1, q, r), \beta^{C'K'}(0, j, a)] \mathrel{+}= \frac{\Delta_F \Delta\theta^K}{4}\, E^y_{jq}\, E^\theta_{ar}$ (Eqs. SI-47b, SI-48b, SI-50b), with $E^y$ (or $E^x$ for `F = 1, 3`) from `get_Ey` (`get_Ex`) and $E^\theta$ from `get_Eth`. The neighboring cells are those returned by `mesh.nhbr_cells_in_nhbr_col` (same-level, parent, or the two children). A face may have two neighboring columns (the neighbor side is finer); each contributes separately.
- **Inflow face on the domain boundary** (no neighbor column): nothing is assembled. See "Inflow Boundary Conditions".

Inflow Boundary Conditions
--------------------------------------------------------------------------------

The publication and the code describe the treatment of the inflow data differently. This section states both without choosing between them.

- **Main text.** The numerical trace (Eq. 12) is $\hat{u} = u^{in}$ on faces with $\vec{s} \cdot \hat{n}_{\partial\Omega_C} < 0$, where "the value of $u^{in}$ comes from either the neighboring spatial h-elements of $C$ or the inflow boundary data $u^{in,bd}$". This is a weak imposition: on a domain-boundary inflow face the data would enter the face integral of Eq. (11a), and the equations of the boundary test functions would stay in the system.
- **SI-13.** Eq. (SI-38) and Table 1 use the neighbor trace only for faces not on $\partial\Omega$ and the element's own trace $u^K$ on faces on $\partial\Omega$; they do not say how $u^{in,bd}$ enters.
- **Code.** The inflow data are imposed strongly at the LGL boundary nodes, by elimination in the linear system:
  1. `dg.matrix.get_intr_mask` marks as boundary DOFs the nodes on each domain-boundary face of a column (no neighbor) for the cells whose quadrant makes that face an inflow face (the "face nodes" above, for all $\theta$ nodes of the cell).
  2. `solve` splits the system matrix into the interior-interior block $A_{II}$ and the interior-boundary block $A_{IB}$; the rows of the boundary DOFs are dropped.
  3. `boundary_conditions_vector` gives the nodal values of $u^{in,bd}$; `solve` keeps their boundary part $g_B$ and solves $A_{II} u_I = F_I - A_{IB} g_B$.
  4. The returned solution takes $u_B = g_B$ exactly at the boundary DOFs (`dg.matrix.merge_vectors`).
  5. The upwind flux of Eq. (12) is assembled only on faces that have a neighboring column (interior and periodic faces). `boundary_convection_matrix` assembles no term on domain-boundary inflow faces.

  The own-trace term that Table 1 of SI-13 prescribes on a domain-boundary inflow face would touch only the rows and columns of the eliminated face nodes, so leaving it out does not change $A_{II}$ or $A_{IB}$. With collocated LGL nodes the code's system is therefore SI-13 with Dirichlet data at the boundary nodes. It is a different discrete problem from the weak imposition through Eq. (12), and the difference matters most for inflow data that are sharply peaked in angle. Neither the main text nor the SI mentions the elimination.

Assembly Order in `solve`
--------------------------------------------------------------------------------

`Problem.solve(mesh, **kwargs)` performs these steps, all collectively on every rank:

1. Initialize `petsc4py` on `MPI.COMM_WORLD` if needed.
2. `check_mesh(mesh)` on the root; the message is broadcast and every rank raises `ValueError` if the mesh is unsupported (PR #26).
3. `mesh.get_ndof()` on the root, broadcast.
4. `ma_sc_mat = self.preconditioner_matrix(mesh, return_preconditioner = False)[0]` ($M_\kappa - M_\sigma$; `kwargs` are not forwarded).
5. `ic_mat = self.interior_convection_matrix(mesh, **kwargs)`.
6. `bc_mat = self.boundary_convection_matrix(mesh, **kwargs)`.
7. `sys_mat = (bc_mat - ic_mat) + ma_sc_mat`.
8. `intr_mask = get_intr_mask(mesh, **kwargs)`; `[A_II, A_IB] = split_matrix(sys_mat, intr_mask)`; `matrix_info = A_II.getInfo()`.
9. `f_vec = self.forcing_vector(mesh, **kwargs)`; keep the interior part $F_I$.
10. `bcs_vec = self.boundary_conditions_vector(mesh, **kwargs)`; keep the boundary part $g_B$.
11. Right-hand side $F_I - A_{IB} g_B$. Initial guess: a copy of $F_I$, used as a nonzero initial guess.
12. Create the KSP with `ksp_type`, operator $A_{II}$, tolerances `rtol`, `atol`, `divtol`, `max_it`, GMRES restart `gmres_restart`, singular-value estimation on, PC `pc_type`; then `ksp.setFromOptions()`, so PETSc options (`-ksp_rtol`, `-ksp_max_it`, `-ksp_monitor`, `PETSC_OPTIONS`, ...) override these settings (PR #28). Solve with convergence history.
13. Read the converged reason, history, iteration count, final residual, and the tolerances actually used (`ksp.getTolerances()`). If the converged reason is negative, the root issues a `RuntimeWarning`; the solution is still returned.
14. `ksp.computeExtremeSingularValues()` is stored in `matrix_info["extreme_singular_values"]`; their ratio (an estimate for the preconditioned operator) is printed when `verbose`.
15. `merge_vectors(u_I, g_B, intr_mask)` gathers the full solution on the root, which converts it into a `Projection`; then `mpi_comm.barrier()`.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `rt` subdirectory are:

- `__init__.py` : Adds the `src` directory to `sys.path` and exports `Problem`.
- `boundary_conditions_vector` : Calculates the vector that holds the inflow boundary data at every node (only the inflow-boundary entries are used by `solve`), including the Dirac-type variant.
- `boundary_convection_matrix` : Calculates the boundary convection (propagation) matrix: the outflow own-trace terms and the upwind inflow terms from neighboring columns.
- `class_Problem` : The `Problem` class stores information about the problem, e.g., the extinction coefficient, scattering coefficient, scattering phase function, forcing term and inflow boundary data, and binds the assembly and solve functions of this directory as its methods.
- `forcing_vector` : Calculates the forcing vector.
- `get_Eth` : Calculates the $\pmb{\mathcal{E}}^{\theta}$ matrices (angular factor of the inflow term) used in the boundary convection matrix.
- `get_Ex` : Calculates the $\pmb{\mathcal{E}}^{x}$ matrices (x-factor of the inflow term on the top and bottom faces) used in the boundary convection matrix.
- `get_Ey` : Calculates the $\pmb{\mathcal{E}}^{y}$ matrices (y-factor of the inflow term on the right and left faces) used in the boundary convection matrix.
- `interior_convection_matrix` : Calculates the interior convection (propagation) matrix.
- `mass_matrix` : Calculates the extinction (mass) matrix.
- `preconditioner_matrix` : Calculates the $extinction - scattering$ matrix in one pass. `solve` uses it as the extinction and scattering part of the system matrix; despite its name it does not return a preconditioner.
- `scattering_matrix` : Calculates the scattering matrix (with the minus sign of Eq. 11a included).
- `solve` : Numerically solves the steady-state monochromatic radiative transfer equation, and checks that the mesh is supported.

Classes and Functions
--------------------------------------------------------------------------------

### `Problem(kappa, sigma, Phi, f, bcs_dirac)` (`class_Problem.py`)

Container for one RTE problem; the assembly and solve functions below are its methods (`problem.solve(mesh)`, `problem.mass_matrix(mesh)`, ...). The constructor only stores its arguments as attributes of the same names.

- `kappa` (callable) : Extinction coefficient $\kappa(x, y)$. Called as `kappa(xxf, yyf)` with `xxf` of shape `[nx, 1]` and `yyf` of shape `[1, ny]`; must broadcast to shape `[nx, ny]` (a scalar return value also works). The type hint in the source lists three arguments; the code calls it with two.
- `sigma` (callable) : Scattering coefficient $\sigma(x, y)$; called like `kappa`. Eq. (3) requires $\kappa > \sigma \ge 0$; this is not checked.
- `Phi` (callable) : Scattering phase function $\Phi(\theta, \theta')$, called with two scalars: the outgoing (test) angle first and the integration angle second. The code does not normalize it; Eq. (2) requires $\int_0^{2\pi} \Phi(\theta, \theta')\, d\theta' = 1$, and the experiment problems supply normalized functions (analytically or by numerical quadrature).
- `f` (callable) : Forcing term $f(x, y, \theta)$, called with three scalars.
- `bcs_dirac` (`list`) : `[bcs, dirac]`. `bcs(x, y, th)` gives the inflow data $u^{in,bd}$ and is called with three scalars at every node of the mesh (not only on the boundary), so it must be defined everywhere; values off the inflow boundary are discarded by `solve`. `dirac = [x_s, y_s, th_s]`, each a `float` or `None`; `[None, None, None]` selects ordinary nodal data (see `boundary_conditions_vector`).

```python
import numpy as np
from dg.mesh import Mesh
from rt import Problem

def kappa(x, y): return 2.0 + 0.0 * x * y
def sigma(x, y): return 0.5 * kappa(x, y)
def Phi(th, phi): return 1.0 / (2.0 * np.pi)          # isotropic, integrates to 1
def f(x, y, th): return 0.0
def bcs(x, y, th): return 1.0 if (x == 0.0 and np.cos(th) > 0.0) else 0.0

problem = Problem(kappa, sigma, Phi, f, [bcs, [None, None, None]])
mesh = Mesh(Ls = [1.0, 1.0], pbcs = [False, False], ndofs = [3, 3, 3], has_th = True)
mesh.ref_mesh(kind = "ang", form = "h")
mesh.ref_mesh(kind = "ang", form = "h")                 # every cell now lies in one quadrant
[uh, info, mat_info] = problem.solve(mesh)               # uh is a Projection on the root
print(info["converged"], info["iteration_number"])
```

### `Problem.solve(mesh, **kwargs)` (`solve.py`)

Assembles and solves the DG system on `mesh` (see "Assembly Order in `solve`").

- `mesh` (`Mesh`) : Spatial-angular mesh with `has_th = True`, the same on every rank. Rejected with `ValueError` (on every rank) when `check_mesh` returns a message.
- `verbose` (`bool`, default `False`) : Print progress and a summary (converged reason, iterations, final residual, condition-number estimate, time). Also forwarded to the assembly methods.
- `ksp_type` (`str`, default `"gmres"`) : PETSc KSP type.
- `pc_type` (`str`, default `"bjacobi"`) : PETSc PC type.
- `rtol`, `atol` (`float`, default `consts.EPS` $= 10^{-15}$) : Relative and absolute residual tolerances.
- `divtol` (`float`, default `sqrt(consts.INF)` $\approx 1.3 \times 10^{154}$) : Divergence tolerance.
- `max_it` (`int`, default `5000`) : Maximum number of iterations.
- `gmres_restart` (`int`, default `50`) : GMRES restart length (ignored by KSP types without restarts).
- Other keyword arguments (e.g., `blocking`) are passed on to the assembly methods except `preconditioner_matrix`.

Before PR #28 the tolerances, `max_it` and the restart length were fixed at these default values; now they are keyword arguments, and the PETSc options database can override all KSP settings.

Returns `[uh_proj, convergence_info, matrix_info]`:

- `uh_proj` : `dg.projection.Projection` of the solution on the root, `None` on other ranks. Inflow-boundary DOFs hold the nodal values of the inflow data exactly.
- `convergence_info` (`dict`) : `converged_reason` (`int`, PETSc `KSPConvergedReason`; negative means not converged), `converged` (`bool`, `converged_reason > 0`), `convergence_history` (`list` of residual norms), `iteration_number`, `res_f` (final residual norm), `res_best` (smallest residual norm in the history), `rtol`, `atol`, `divtol`, `max_it` (values in effect after `setFromOptions`), `gmres_restart` (the keyword value).
- `matrix_info` (`dict`) : `A_II.getInfo()` of the interior-interior block (`nz_used`, `nz_allocated`, `memory`, `mallocs`, ...) plus `extreme_singular_values = [emax, emin]` from the Krylov iteration.

Side effects: MPI collectives throughout; a `RuntimeWarning` on the root when the solve does not converge; module-level caches in `get_Ex`, `get_Ey`, `interior_convection_matrix` may grow.

### `check_mesh(mesh)` (`solve.py`)

Returns an error message (`str`) if `mesh` is outside the assumptions of the discretization, and `None` otherwise (PR #26). It checks every leaf column for:

- `min(col.ndofs) < 2`: the inflow condition is imposed at the end LGL nodes, so each spatial direction needs at least 2 nodes;
- when `mesh.has_th`, any leaf cell with `quadrant is None`: the upwind fluxes and the inflow mask use the quadrant of each cell, so the mesh must be refined in angle at least twice.

It does not check `has_th` itself. A mesh with `has_th = False` has a single cell of zero angular width per column, so every matrix entry is zero; `solve` is meant for `has_th = True`. `check_mesh` only reads the mesh and does no communication; `solve` calls it on the root.

### `Problem.mass_matrix(mesh, **kwargs)` (`mass_matrix.py`)

Assembles the extinction matrix $M_\kappa$ (diagonal; Eq. SI-28).

- `mesh` (`Mesh`) : The mesh.
- `verbose` (`bool`, default `False`) : Print progress and timing.
- `blocking` (`bool`, default `True`) : Call `MPI_comm.Barrier()` before returning.
- Returns : `PETSc.Mat` of size `[ndof, ndof]`.

Not called by `solve` (which uses `preconditioner_matrix`); used by `experiments/test-rt`.

`calc_col_matrix(mesh, col_key, kappa)` returns the `scipy.sparse` CSR block of column `col_key` (block diagonal over its cells, square of size `nx * ny * sum(nth)`), or `None` for a non-leaf column.

### `Problem.scattering_matrix(mesh, **kwargs)` (`scattering_matrix.py`)

Assembles the scattering matrix $-M_\sigma$ (Eq. SI-34), with the minus sign of Eq. (11a) included, so that $M_\kappa + (\text{this}) = $ `preconditioner_matrix`. Keyword arguments, MPI behavior and return type are as for `mass_matrix`. Its verbose message reads "Constructed (Extinction - Scattering) Matrix", copied from `preconditioner_matrix`.

`calc_col_matrix(mesh, col_key, sigma, Phi)` returns the CSR block of column `col_key`: an `ncells x ncells` block matrix whose block `(K, K')` has the entries above. Entries with $|value| \le 10\,\epsilon = 10^{-14}$ are dropped, and in off-diagonal blocks whole spatial nodes with $|w^x_i w^y_j \sigma(x_i, y_j)| \le 10^{-14}$ are skipped. The threshold is absolute (not scaled by the element size). For a non-leaf column it returns `[None, None]`.

### `Problem.preconditioner_matrix(mesh, **kwargs)` (`preconditioner_matrix.py`)

Assembles $M_\kappa - M_\sigma$ in one pass (diagonal blocks add the extinction term on the $\theta$-diagonal, $r = a$). `solve` uses it for the extinction and scattering part of the system matrix.

- `return_preconditioner` (`bool`, default `False`) : When `True`, each rank also inverts its column blocks with `scipy.sparse.linalg.inv`; the inverses are discarded.
- `verbose`, `blocking` : As for `mass_matrix`.
- Returns : `[M, None]`, where `M` is a `PETSc.Mat` of size `[ndof, ndof]`. The second element is always `None`, although the docstring says the preconditioner is returned.

`calc_col_matrix(mesh, col_key, kappa, sigma, Phi, return_preconditioner = False)` returns `[col_mtx, inv_col_mtx]` (CSR block and its inverse or `None`; `[None, None]` for a non-leaf column), with the same thresholds as the scattering matrix.

### `Problem.interior_convection_matrix(mesh, **kwargs)` (`interior_convection_matrix.py`)

Assembles the interior convection matrix $I$, the volume term $\int \vec{s} \cdot \nabla_{\vec{x}} v\, u$ of Eq. (11a) (Eq. SI-37). `solve` subtracts it. Keyword arguments, MPI behavior and return type are as for `mass_matrix`.

`calc_col_matrix(mesh, col_key)` returns the CSR block of column `col_key` (block diagonal over cells; `None` for a non-leaf column). The reference derivative matrices `ddx_phi[p, i]` $= \bar\phi'_p(\bar x_i)$ and `ddy_psi[q, j]` are cached in the module-level dictionaries `ddx_phis` and `ddy_psis`, keyed by the number of nodes.

### `Problem.boundary_convection_matrix(mesh, **kwargs)` (`boundary_convection_matrix.py`)

Assembles the boundary convection matrix $B$, the face term $\int_{\partial\Omega_C} (\vec{s} \cdot \hat{n})\, \hat{u}\, v$ of Eq. (11a) with the upwind trace (see "The Discrete System" and "Inflow Boundary Conditions").

- `verbose`, `blocking` : As for `mass_matrix`.
- Returns : `PETSc.Mat` of size `[ndof, ndof]`.

MPI behavior differs from the other methods: the root builds the whole matrix as a `scipy.sparse` block matrix (column blocks `M^{CC}` and `M^{CC'}`), then sends each other rank the rows of its PETSc ownership range (`MPI_comm.send` / `MPI_comm.recv(source = 0)`), and every rank inserts its rows. The root therefore holds the full boundary matrix in memory.

`calc_col_matrix(mesh, col_key_0, col_face_0)` builds the contributions of face `col_face_0` of column `col_key_0` and returns `[col_mtx_00, [col_key_1, col_mtx_01], [col_key_2, col_mtx_02]]`: the COO block coupling the column with itself (outflow own-trace terms), and for each of the up to two neighboring columns through the face, its key and the COO block coupling the column to it (inflow terms); a missing or duplicate neighbor gives `[None, None]`. Entries with $|value| \le 10^{-15}$ are dropped. Face nodes use the LGL end nodes, so `nx, ny >= 2` is required.

`Theta_F(theta, col_face)` returns $\cos(\theta - F\pi/2) = \vec{s} \cdot \hat{n}_F$ (Eq. SI-41) for an array `theta`.

### `get_Ex(mesh, col_key_0, col_key_1)` and `get_Ey(mesh, col_key_0, col_key_1)` (`get_Ex.py`, `get_Ey.py`)

Spatial factor of the inflow term between column `col_key_0` ($C$, the column being assembled) and its neighbor `col_key_1` ($C'$, the upwind column): $E^x$ for the top and bottom faces (overlap in $x$), $E^y$ for the right and left faces (overlap in $y$). `get_Ey` is `get_Ex` with $y$, `ndofs[1]` and `pos[1], pos[3]` in place of $x$, `ndofs[0]` and `pos[0], pos[2]`.

- Returns : `np.ndarray` of shape `[n_1, n_0]` (`n` the number of nodes in $x$, resp. $y$), with `E[i, p]` $\approx \frac{2}{\Delta x^C} \int \phi^{C'}_i \phi^C_p\, dx$ over the overlap; `boundary_convection_matrix` multiplies by $\Delta x^C / 2$.
- The quadrature follows the four cases of Eq. (SI-45)/(SI-49): with $A$ the element with more nodes and $B$ the element with the smaller interval ($A = B$ for equal intervals), the rule of $A$ is mapped onto the interval of $B$. When $C'$ is finer, the result carries the factor $1/2 = \Delta x^{C'} / \Delta x^C$.
- The neighbor relation is `(lv_0 - lv_1, pos)` with `pos` in `"s"` (same level), `"l"`, `"u"` (lower or upper half, from the midpoints). Only level differences of $-1, 0, 1$ are handled (2:1 balance); a larger difference leaves `pos` undefined and raises an exception.
- Results are cached in the module-level dictionaries `Ex_matrices` / `Ey_matrices` keyed by `(n_0, n_1, (lv_0 - lv_1, pos))`; the cache persists across meshes within a process.

`get_Ex_old` and `get_Ey_old` are earlier versions with the same signature and return value. They are not called anywhere in the repository (they read the same cache but do not write to it). `get_Ey.py` also imports `dg.projection.get_f2f_matrix`, which it does not use.

### `get_Eth(mesh, col_key_0, cell_key_0, col_key_1, cell_key_1, F)` (`get_Eth.py`)

Angular factor of the inflow term on face `F` of column `col_key_0` from the neighboring cell `cell_key_1` in column `col_key_1`.

- `col_key_0`, `cell_key_0` : The element $(C, K)$ being assembled (test functions).
- `col_key_1`, `cell_key_1` : The neighboring (upwind) element $(C', K')$ (trial functions).
- `F` (`int`) : Face of $C$, `0` to `3`.
- Returns : `np.ndarray` of shape `[nth_1, nth_0]` with

  `E_th[a, r] = share_r * w_r * Theta_F(th_r) * xi'_a(th_r)`,

  where $\theta_r$, $w_r$ are the LG nodes and reference weights of $K$, $\xi'_a$ is the $a$-th basis function of $K'$, and `share_r` is $1$ for a node of $K$ strictly inside $K'$, $1/2$ for a node on the boundary of $K'$ (the midpoint node of $K$ for an odd number of nodes and two finer neighbors), and $0$ for a node outside $K'$. Entries with magnitude below $10^{-15}$ are set to zero. `boundary_convection_matrix` multiplies by $\Delta\theta^K / 2$. Not cached, because $\Theta_F$ depends on the cell position.

Since PR #24 the integral is always collocated at the nodes of $K$ (the "own-node" rule), the same quadrature used for the volume and outflow terms of $K$, so that all angular integrals of $K$ share one quadrature and a constant state is preserved on non-conforming angular meshes. This corresponds to the case $A = B = K$ of Eq. (SI-51) for every neighbor pair. Eq. (SI-51) instead selects among four rules by which element has more nodes ($A$) and which has the smaller interval ($B$); the code differs from it when $K'$ has more nodes or a smaller interval than $K$.

`Theta_F_func(theta, F)` returns $\cos(\theta - F\pi/2)$.

### `Problem.forcing_vector(mesh, **kwargs)` (`forcing_vector.py`)

Assembles $F$ with $F_{\beta(i,j,a)} = \frac{\Delta x \Delta y \Delta\theta}{8} w^x_i w^y_j w^\theta_a f(x_i, y_j, \theta_a)$ (Eq. 11b by quadrature).

- `verbose`, `blocking` : As for `mass_matrix`. `precondition` (default `False`) is accepted and unused.
- Returns : `PETSc.Vec` of size `ndof` (the type hint says `np.ndarray`). Only nonzero entries are set.

`calc_col_vec(mesh, col_key, f)` returns the `np.ndarray` block of column `col_key` (length `nx * ny * sum(nth)`).

### `Problem.boundary_conditions_vector(mesh, **kwargs)` (`boundary_conditions_vector.py`)

Assembles the vector $g$ of inflow data in the global DOF order, at every node of the mesh. `solve` keeps only the entries at the inflow-boundary DOFs of `get_intr_mask`.

- `verbose`, `blocking` : As for `mass_matrix`.
- Returns : `PETSc.Vec` of size `ndof`. Only nonzero entries are set.
- MPI: starts with `mesh = MPI_comm.bcast(mesh, root = COMM_ROOT)`, so every rank works on a copy of the root's mesh (collective; the caller's object is not changed).

`calc_col_vec(mesh, col_key, bcs_dirac)` returns the `np.ndarray` block of column `col_key`. With `bcs_dirac = [bcs, dirac]`:

- If every entry of `dirac` is `None`, the entry of node $(i, j, a)$ is `bcs(x_i, y_j, th_a)`.
- Otherwise (Dirac-type data, PR #27), an element receives data only if it contains every given coordinate (closed intervals, so a coordinate on an element face belongs to both adjacent elements); all other elements get zeros in their own slots. In an element that receives data, the entry of node $(i, j, a)$ is `bcs(X, Y, T) * P_i * Q_j * R_a`, where for each given coordinate, e.g. `x_s`, `X = x_s` and `P_i = max(0, l_i(pull_back(x0, x1, x_s)))` (the $i$-th Lagrange basis function at the source, with negative values clipped), and for each `None` coordinate `X = x_i` and `P_i = 1`. A coordinate equal to `0.0` counts as given.

Before PR #27, the presence of Dirac data was tested with `any(dirac)` (so a coordinate `0.0` was treated as absent), data reached every element for which any dimension passed the containment test (dimensions without a coordinate always pass), and the write index was not advanced for cells without data, so later cells wrote into the wrong slots.

Code and Paper Differences
--------------------------------------------------------------------------------

Collected from the sections above, stated without preference:

- Inflow data: weak imposition through Eq. (12) in the main text; own trace on $\partial\Omega$ in SI-13 (Eq. SI-38, Table 1); strong imposition by elimination at the LGL boundary nodes in the code ("Inflow Boundary Conditions").
- Angular inflow factor: Eq. (SI-51) chooses the quadrature by node count and interval size; `get_Eth` always uses the nodes of $K$ (PR #24).
- Local DOF order: Eqs. (22a, 22b) of SI-9 put $x$ fastest; the code puts $\theta$ fastest.
- `Problem` stores whatever `Phi` it is given; Eq. (2) assumes it is normalized.

Subdirectory Structure
--------------------------------------------------------------------------------

The `rt` subdirectory has no subdirectories.
