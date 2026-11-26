`error`
================================================================================

This subdirectory contains the source code regarding calculating various kinds of error. Each function here is attached to `Error_Indicator` as a method (see `class_Error_Indicator.py`), so `self` is an `Error_Indicator` and the functions are called as `err_ind.error_angular_jump()` and so on.

How It Fits Into the Package
--------------------------------------------------------------------------------

After `rt.Problem.solve` returns $u_{hp}$ and an `Error_Indicator` is built from it, exactly one of these methods is called. Each method

1. computes a column indicator for every leaf column and a cell indicator for every leaf cell of `self.proj.mesh`, and stores them in a new `self.cols` (`dict[int, Error_Indicator_Column]`, each holding `cells`, a `dict[int, Error_Indicator_Cell]`), replacing any previous values;
2. sets `self.error`, `self.col_max_error`, `self.cell_max_error`, and `self.error_to_resolve = 0`;
3. marks elements against the thresholds of Eq. (15) and, for `ref_form = "hp"`, calls `self.col_hp_steer` and `self.cell_hp_steer` (the marking block is described in `../README.md`). `self.error_to_resolve` becomes the sum of the marked indicators.

`Error_Indicator.ref_by_ind` then uses the marks. The angular-jump and spatial-jump indicators are the error indicators of the paper's algorithm (Section 3.1); the experiment drivers use `error_angular_jump` (and, in the two-strong-scatterers experiment, `error_spatial_jump`) to drive refinement. The analytic and high-resolution errors measure the accuracy of $u_{hp}$ for the error histories, and the random error is a test device.

These methods call `dg.quadrature.quad_xyth` (LGL nodes in $x$, $y$; LG nodes in $\theta$; ascending on $[-1, 1]$), `dg.quadrature.lag_eval` (Lagrange basis evaluation), and `dg.projection.push_forward` / `pull_back` (maps between $[-1, 1]$ and an element). Nodal values are `cell.vals` of shape `[nx, ny, nth]`.

Summary of the Indicators
--------------------------------------------------------------------------------

Write $e_K$ for the integral of a squared difference over one spatial-angular element $(C, K)$.

| Method | Cell indicator | Column indicator | `self.error` | MPI |
| --- | --- | --- | --- | --- |
| `error_analytic` | $\sqrt{e_K / \Vert u \Vert^2}$ | $\sqrt{\sum_{K \in C} e_K / \Vert u \Vert^2}$ | $\sqrt{\sum e_K / \Vert u \Vert^2}$ (relative $L^2$ error) | none |
| `error_angular_jump` | $\eta_{\vec{s}}^{(C,K)}$, Eq. (14) | $\sqrt{\sum_{K \in C} (\eta_{\vec{s}}^{(C,K)})^2}$ | $\sqrt{\sum_{(C,K)} (\eta_{\vec{s}}^{(C,K)})^2}$ | none |
| `error_spatial_jump` | `-consts.INF` (placeholder) | $\eta_{\vec{x}}^{C}$, Eq. (13) | $\sqrt{\sum_C (\eta_{\vec{x}}^{C})^2}$ | none |
| `error_high_resolution` | $\sqrt{e_K / \Vert u_{hr} \Vert^2}$ | $\sqrt{\sum_{K \in C} e_K / \Vert u_{hr} \Vert^2}$ | $\sqrt{\sum e_K / \Vert u_{hr} \Vert^2}$ | collective |
| `error_random` | uniform on $[0, 1)$ | sum of its cell values | sum of all cell values | none |

The jump indicators are absolute (not divided by a norm of the solution). The analytic and high-resolution errors are relative.

`error_analytic`
--------------------------------------------------------------------------------

```python
Error_Indicator.error_analytic(self, anl_sol, quad_rtol: float = 1.e-2,
                               max_quad_dbls: int = 4) -> None
```

Relative $L^2$ error of $u_{hp}$ against an analytic solution $u$, per cell, per column, and over the mesh.

Arguments:

- `anl_sol` (callable `(x, y, th) -> array`) : the analytic solution in physical coordinates, $x \in [0, L_x]$, $y \in [0, L_y]$, $\theta \in [0, 2\pi]$. It must broadcast: it is called with `x` of shape `[nx', 1, 1]`, `y` of shape `[1, ny', 1]`, `th` of shape `[1, 1, nth']` and must return shape `[nx', ny', nth']`. It is also called with scalar floats by `scipy.integrate.nquad`.
- `quad_rtol` (`float`, default `1.e-2`) : relative tolerance for the change of the global error between quadrature doublings.
- `max_quad_dbls` (`int`, default `4`) : maximum number of quadrature doublings.

Steps:

1. $\Vert u \Vert^2 = \int_0^{L_x} \int_0^{L_y} \int_0^{2\pi} u^2$ with `scipy.integrate.nquad` (default tolerances), recomputed on every call.
2. $e_K = \int_{C \times K} (u - u_{hp})^2$ with `quad_error` (below) at `hr_mult = 1`, then `hr_mult = 2, 4, ...` until the global relative error changes by at most `quad_rtol * err + 1.e-12`, or `max_quad_dbls` doublings have been made (so at most `max_quad_dbls + 1` evaluations). If it does not converge, `warnings.warn` is called and the last values are used.
3. Store the indicators of the table above, then mark and steer.

Returns `None`. Side effects: sets `self.cols`, `self.error`, `self.col_max_error`, `self.cell_max_error`, `self.error_to_resolve` and the marks; fills the module-level caches `phi_projs`, `psi_projs`, `xsi_projs` (interpolation matrices keyed by `(n, n_hr)`), which persist for the life of the process; may issue a `UserWarning`.

Example (adapted from `tests/test_amr/test_error_analytic.py`):

```python
def u(x, y, th):   # broadcasts over [nx', 1, 1], [1, ny', 1], [1, 1, nth']
    return np.exp(-((x - 1.) / 3.)**2) * np.exp(-2. * (2. - y)) \
        * np.exp(-((96. / (2. * np.pi)) * (th - 7. * np.pi / 5.))**2)

proj = Projection(mesh, u)   # nodal interpolant of u on mesh
err_ind = Error_Indicator(proj, ref_kind = "ang", ref_form = "h", ref_tol = [0., 0.])
err_ind.error_analytic(u)
print(err_ind.error)         # relative L2 error
```

`quad_error`
--------------------------------------------------------------------------------

```python
quad_error(self, anl_sol, hr_mult: int) -> list
```

Module-level helper in `error_analytic.py` (not exported and not a method, but takes the `Error_Indicator` as `self`). For each leaf cell it interpolates $u_{hp}$ from its nodes to `hr_mult * (n + 3)` nodes per direction (LGL in $x$ and $y$, LG in $\theta$) with `numpy.einsum`, evaluates `anl_sol` at the same nodes, and returns

- `col_errs` (`dict`) : `{col_key: [col_err, {cell_key: cell_err}]}`, where `cell_err` $= \frac{\Delta x \Delta y \Delta\theta}{8} \sum w_p w_q w_r (u - u_{hp})^2$ and `col_err` is the sum over the column's cells;
- `mesh_err` (`float`) : the sum over all cells.

All values are squared and not normalized.

`error_angular_jump`
--------------------------------------------------------------------------------

```python
Error_Indicator.error_angular_jump(self) -> None
```

Angular-jump error indicator, Eq. (14) of the paper.

1. For every leaf cell, $u_{hp}$ is evaluated at the two angular ends of the cell (reference $\theta = -1$ and $\theta = +1$) by Lagrange interpolation in $\theta$. This gives two arrays of shape `[nx, ny]` at the column's spatial nodes.
2. For a cell $K$ with lower neighbor `cell.nhbr_keys[0]` and upper neighbor `cell.nhbr_keys[1]` (both in the same column; the angular mesh is periodic, so the cell at $\theta = 0$ is compared with the cell ending at $2\pi$, and a column with one cell compares the cell with itself):
   $e_K = \frac{1}{4} \sum_{i,j} w_i w_j \cdot \frac{1}{2} \left[ (u_K(\theta_0) - u_{K^-}(\theta_1))^2 + (u_K(\theta_1) - u_{K^+}(\theta_0))^2 \right]_{ij}$.
   Since $\int_{\Omega_C} f \, d\vec{x} = \frac{|\Omega_C|}{4} \sum w_i w_j f_{ij}$, this equals $\frac{1}{|\partial S^K| \, |\Omega_C|} \int_{\partial S^K} \int_{\Omega_C} [\![u_{hp}]\!]_{\vec{s}}^2$ with $|\partial S^K| = 2$, the square of Eq. (14).
3. Cell indicator $\sqrt{e_K}$, column indicator $\sqrt{\sum_{K \in C} e_K}$, `self.error` $= \sqrt{\sum e_K}$; then mark and steer.

Returns `None`. Side effects: as for `error_analytic`, without caches.

`error_spatial_jump`
--------------------------------------------------------------------------------

```python
Error_Indicator.error_spatial_jump(self) -> None
```

Spatial-jump error indicator, Eq. (13) of the paper. It produces column indicators only.

For each leaf column $C$, faces are numbered counterclockwise from the right: `0` is $x = x_1$, `1` is $y = y_1$, `2` is $x = x_0$, `3` is $y = y_0$. For each face and each distinct neighbor column $C'$ in `col.nhbr_keys[face]` (a `None` key, i.e. a non-periodic domain boundary, is skipped; with periodic boundaries a column can be its own neighbor), and for each cell $K$ of $C$ and each cell $K'$ of $C'$ that overlaps it in angle (`Mesh.nhbr_cells_in_nhbr_col`):

- the traces of $u_{hp}$ on the face (`vals[-1,:,:]`, `vals[:,-1,:]`, `vals[0,:,:]`, `vals[:,0,:]` for faces 0-3, and the opposite slice for $C'$) are interpolated to a common grid on the shorter of the two faces and the shorter of the two angular intervals, using $\max(n_C, n_{C'})$ LGL nodes along the face and $\max(n_\theta, n'_\theta)$ LG nodes in $\theta$;
- $J_C$ += $\frac{\Delta s \, \Delta\theta}{4} \sum w w_\theta (u_C - u_{C'})^2$, where $\Delta s$ and $\Delta\theta$ are the lengths of the common face segment and angular interval.

The column indicator is $\eta_{\vec{x}}^{C} = \sqrt{J_C / (|\partial\Omega_C| \cdot 2\pi)}$ with $|\partial\Omega_C| = 2\Delta x + 2\Delta y$, the full perimeter including faces on the domain boundary (where no jump is added). `self.error` $= \sqrt{\sum_C (\eta_{\vec{x}}^{C})^2}$.

Every cell indicator is set to the placeholder `-consts.INF` (the most negative finite `REAL`) and `self.cell_max_error = -consts.INF`. Use this method with `ref_kind = "spt"`. With `"ang"` or `"all"`, the angular threshold is `ang_ref_tol * -consts.INF`: no cell is marked when `ang_ref_tol < 1`, and every cell is marked (adding `-consts.INF` to `error_to_resolve`) when `ang_ref_tol = 1`.

Returns `None`. The projection onto the common grid uses four nested Python loops per pair of cells, so the cost grows quickly with the number of degrees of freedom.

`error_high_resolution`
--------------------------------------------------------------------------------

```python
Error_Indicator.error_high_resolution(self, problem: Problem, uh_hr: Projection = None,
                                      **kwargs) -> list
```

Relative $L^2$ difference between $u_{hp}$ and a solution $u_{hr}$ on a *p*-enriched copy of the mesh. This function is an MPI collective: every rank of `MPI.COMM_WORLD` must call it (it initializes `petsc4py` if needed).

Arguments:

- `problem` (`rt.Problem`) : the problem used to compute $u_{hr}$.
- `uh_hr` (`dg.projection.Projection` or `None`, default `None`) : a precomputed high-resolution solution (needed on the root rank only). It must have the same column and cell keys and positions as `self.proj.mesh`, differing only in `ndofs`. If `None`, it is computed.
- `kwargs` : `ang_ref_offset` (default `0`) and `spt_ref_offset` (default `0`), the number of degrees of freedom added to every leaf cell in $\theta$ and to every leaf column in each of $x$ and $y$; the linear-solver settings `ksp_type` (default `"gmres"`), `pc_type` (default `"bjacobi"`) and, if given, `rtol`, `atol`, `divtol`, `max_it`, `gmres_restart`, which are forwarded to `problem.solve`. `verbose` (default `False`) and `blocking` (default `True`) are accepted but not used; the final barrier always runs. Other keys are ignored, so the experiment drivers pass `**solver_params, **hr_err_params` from `input.json`.

Steps:

1. If `uh_hr` is `None`: the root rank deep-copies `self.proj.mesh` and adds the offsets to `ndofs` (a uniform offset; `Mesh.ref_mesh(form = "p")` is not used because its $|\Delta n| \le 1$ cascade would refine some elements more than once). The mesh is broadcast, `problem.solve(mesh_hr, **solver_kwargs)` runs on all ranks, and `PETSc.garbage_cleanup` is called.
2. On the root rank: $\Vert u_{hr} \Vert^2$ by quadrature on $u_{hr}$'s nodes; for each cell, $u_{hp}$ is interpolated to $u_{hr}$'s nodes and $e_K = \frac{\Delta x \Delta y \Delta\theta}{8} \sum w w w (u_{hr} - u_{hp})^2$ with $u_{hr}$'s LGL/LG weights. Then the indicators of the table above are stored, and elements are marked and steered.
3. `mpi_comm.barrier()`.

Returns `[uh_hr, info, mat_info]`: the high-resolution `Projection` (on the root rank; `Problem.solve` returns `None` on other ranks), and the convergence-information and matrix-information dictionaries returned by `Problem.solve` (both `None` when `uh_hr` was supplied).

Side effects: on the root rank only, sets `self.cols`, `self.error`, `self.col_max_error`, `self.cell_max_error`, `self.error_to_resolve` and the marks; on other ranks these attributes are not set. Fills this module's caches `phi_projs`, `psi_projs`, `xsi_projs`.

Notes: the $n_{hr}$-point LGL rule integrates polynomials up to degree $2n_{hr} - 3$ exactly, while $(u_{hr} - u_{hp})^2$ has degree $2n_{hr} - 2$ in $x$ and $y$, so $e_K$ is a quadrature approximation. The interpolation to $u_{hr}$'s nodes uses six nested Python loops per cell. With both offsets 0, $u_{hr}$ is computed on the same mesh and the result is the difference between two solves.

Example (from `experiments/manufactured-solution/experiment.py`, called on all ranks):

```python
err_ind_hr = Error_Indicator(uh, **ref_strat)
[uh_hr, info_hr, mat_info_hr] = err_ind_hr.error_high_resolution(
    problem, **solver_params, **{"ang_ref_offset": 3, "spt_ref_offset": 0})
```

`error_random`
--------------------------------------------------------------------------------

```python
Error_Indicator.error_random(self, rng: np.random.Generator = None) -> None
```

Assigns each leaf cell an indicator drawn from `rng.random()` (uniform on $[0, 1)$). Each column indicator is the sum of its cell indicators, and `self.error` is the sum over all cells (sums, not root-sum-of-squares). Then marks and steers.

- `rng` (`numpy.random.Generator` or `None`, default `None`) : the random number generator; if `None`, an unseeded `np.random.default_rng()` is used. Pass a seeded generator for reproducible results.

Returns `None`. Used to exercise the marking, steering and refinement code without a meaningful error.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `error` subdirectory are:

- `__init__.py` : Exports `error_analytic`, `error_angular_jump`, `error_spatial_jump`, `error_high_resolution`, and `error_random`.
- `error_analytic.py` : Calculates the relative $L_2$ error between a given numerical and analytic solution, per cell and per column, with adaptive quadrature (`error_analytic`, helper `quad_error`).
- `error_angular_jump.py` : Calculates the angular-jump error indicator (Eq. 14) for a given numerical solution.
- `error_high_resolution.py` : Calculates the relative $L_2$ difference between a given numerical solution and a high-resolution (*p*-enriched) solution of the same `Problem`.
- `error_random.py` : Randomly assigns error to each spatial and angular element in an *hp*-mesh for a given numerical solution.
- `error_spatial_jump.py` : Calculates the spatial-jump error indicator (Eq. 13) for a given numerical solution.
