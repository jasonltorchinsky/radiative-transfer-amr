`test_rt`
================================================================================

This subdirectory contains regression tests for the code in `src/rt`, the `Problem` class that assembles the DG system of Eqs. 10-12 on a `dg.mesh.Mesh` and solves it with PETSc (`Problem.solve`). In the experiment drivers, `Problem.solve` is the collective step of each trial: every rank calls it, and it returns the solution as a `dg.projection.Projection` on the root rank (`None` on the others) together with the Krylov convergence information.

Tests that call `Problem.solve` or `Error_Indicator.error_high_resolution` are collective: under `mpirun -n N python -m pytest --with-mpi` they run one distributed solve across the ranks, and they check the solution only on the root rank (`consts.COMM_ROOT` = 0). The other tests run independently on each rank.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `test_rt` subdirectory are:

- `test_boundary_conditions_vector.py` : Issue #13 (PR #27), Dirac boundary data in `calc_col_vec`.
- `test_get_Eth.py` : Issue #10 (PR #24), the angular factor of the upwind inflow term on non-conforming angular faces.
- `test_solve.py` : Issue #12 (PR #26), `Problem.solve` rejects unsupported meshes.
- `test_solve_krylov.py` : Issue #14 (PR #28), configurable Krylov settings and reporting of unconverged solves.

`test_boundary_conditions_vector.py`
--------------------------------------------------------------------------------

Regression test for issue #13. A `Problem` takes its boundary data as `bcs_dirac = [bcs, [x_s, y_s, th_s]]`; a coordinate that is not `None` makes the data a Dirac delta in that coordinate (a point source, a line source, or a beam). `rt.boundary_conditions_vector.calc_col_vec(mesh, col_key, bcs_dirac)` returns the boundary-data vector of one column, ordered cell by cell (sorted cell keys) and, within a cell, with index order `[i, j, a]` ($a$ fastest). The data must reach only the elements that contain every given coordinate, at those elements' own offsets in the column vector, and a coordinate of `0.0` must count as a location (before the fix it was treated like `None`).

Module-level helpers:

- `make_mesh() -> Mesh` : $2 \times 2$ columns on $[0, 1]^2$, four cells per column (one per quadrant), two nodes in each dimension (`ndofs = [2, 2, 2]`).
- `g(x, y, th)` : the boundary function, identically 1.
- `cell_blocks(mesh, col_key, dirac) -> list` : calls `calc_col_vec(mesh, col_key, [g, dirac])`, splits the result into `(column, cell, block)` triples of length $n_x n_y n_\theta$, and asserts that the blocks cover the whole vector.
- `all_blocks(mesh, dirac) -> list` : `cell_blocks` for all columns.

Tests:

- `test_point_source_only_in_its_element` : point source at $(x, y, \theta) = (0.25, 0.25, 7\pi/4)$, the midpoint of one element. Exactly one block is non-zero, and all its entries equal $1/8$ (the product of three linear Lagrange basis functions, each $1/2$ at the midpoint).
- `test_line_source_only_in_columns_containing_it` : line source $x = 0.25$. Blocks of the two columns with $0 < x < 0.5$ equal $1/2$; all others are zero.
- `test_zero_coordinate_is_a_dirac_location` : beam at $\theta = 0$. Only the cells with $\theta_0 = 0$ are non-zero, and in each exactly half of the entries are non-zero: `calc_col_vec` clips negative basis-function values to zero (`max(0, lag_eval(...))`), and with two Gauss nodes one angular basis function is negative at $\hat\theta = -1$. The test records this behavior; the block is therefore not the nodal interpolant of $g$.

`test_get_Eth.py`
--------------------------------------------------------------------------------

Regression test for issue #10. `rt.get_Eth.get_Eth(mesh, col_key_0, cell_key_0, col_key_1, cell_key_1, F)` returns `E_th[aa, rr]`, shape `[nth_1, nth_0]`: the angular factor of the upwind inflow term on face `F` of column `col_key_0`, cell $K$ (`cell_key_0`), from neighboring cell $K''$ (`cell_key_1`) of column `col_key_1`. After the fix it is collocated at the Legendre-Gauss nodes $\theta_r$ of the receiving cell $K$: `E_th[aa, rr]` $= s_r \, w_r \, \Theta_F(\theta_r) \, \xi''_a(\theta_r)$, where $\Theta_F(\theta) = \cos(\theta - F\pi/2)$ (`Theta_F_func`) and $s_r$ is 1 if $\theta_r$ is inside $K''$, $1/2$ if it is on the boundary of $K''$, and 0 otherwise. The same rule is used for the volume and outflow terms of $K$, so a constant state is preserved on non-conforming angular meshes.

Module-level helper:

- `make_mesh(form, ndof_th) -> Mesh` : $2 \times 2$ columns on $[0, 2] \times [0, 1]$ with `ndofs = [3, 3, ndof_th]` and four cells per column; the first cell of the first column is then refined in angle with `form` (`"h"` or `"p"`), so the angular faces between this column and its neighbors are non-conforming.

Tests:

- `test_get_Eth_constant_trace[form, ndof_th]`, `form` in `"h"`, `"p"` and `ndof_th` in 3, 4 : for every column, face, neighboring column and cell, the sum over $a$ and over all neighboring cells $K''$ of `E_th[a, r]` must equal $w_r \Theta_F(\theta_r)$, the outflow factor of $K$ (to $10^{-14}$). This is the statement that a constant neighbor trace produces the same inflow as the cell's own trace.
- `test_constant_state_non_conforming_angular_mesh[form, scatter]` : solves a problem whose exact solution is the constant $u = c = 2$ ($\kappa = 1 + x/2$; $\sigma = 0$ or $\kappa/2$; isotropic $\Phi = 1/(2\pi)$; $f = (\kappa - \sigma) c$; inflow data $c$) on `make_mesh(form, 3)` with `Problem.solve(mesh, ksp_type = "gmres", pc_type = "lu")`, and asserts on the root rank that every nodal value equals $c$ to $10^{-12}$. Collective.

`test_solve.py`
--------------------------------------------------------------------------------

Regression test for issue #12. `Problem.solve` calls `rt.solve.check_mesh(mesh)` on the root rank, broadcasts the result, and raises `ValueError` on every rank if the mesh is outside the assumptions of the discretization: a leaf cell spanning more than one angular quadrant (`cell.quadrant is None`; the upwind fluxes and the inflow condition use the quadrant), or a leaf column with fewer than 2 LGL nodes in a direction (the inflow condition is imposed at the end nodes).

Module-level helpers:

- `constant_problem() -> Problem` : $\kappa = 1$, $\sigma = 0$, $f = 2$, inflow value 2, so $u = 2$.
- `make_mesh(ndofs, nref_ang) -> Mesh` : $[0, 1]^2$ refined `nref_ang` times in angle and once in space.

Tests:

- `test_solve_rejects_quadrant_spanning_cells[nref_ang]`, `nref_ang` in 0, 1 : with one or two angular cells, `solve` raises `ValueError` matching `"quadrant"`.
- `test_solve_rejects_one_spatial_node` : with `ndofs = [1, 1, 2]`, `solve` raises `ValueError` matching `"LGL nodes"`.
- `test_solve_accepts_supported_mesh` : with `ndofs = [3, 3, 3]` and two angular refinements, `solve` runs and, on the root rank, the solution equals 2 to $10^{-8}$. Collective.

`test_solve_krylov.py`
--------------------------------------------------------------------------------

Regression tests for issue #14. `Problem.solve(mesh, **kwargs)` accepts `ksp_type`, `pc_type`, `rtol`, `atol`, `divtol`, `max_it` and `gmres_restart` (defaults `"gmres"`, `"bjacobi"`, `consts.EPS`, `consts.EPS`, $\sqrt{\texttt{consts.INF}}$, 5000, 50, the values that were hard-coded before), then calls `ksp.setFromOptions()` so that the PETSc options database (command line, `PETSC_OPTIONS`) can override them. It returns `[uh, convergence_info, matrix_info]`, where `convergence_info` (on every rank) has the keys `converged_reason`, `converged`, `convergence_history`, `iteration_number`, `res_f`, `res_best`, `rtol`, `atol`, `divtol`, `max_it` and `gmres_restart`. If the solve does not converge (`converged_reason` $\leq 0$), the root rank issues a `RuntimeWarning` containing `"did not converge"`.

Module-level helpers:

- `get_problem_and_mesh() -> [Problem, Mesh]` : $\kappa = 10$, $\sigma = 1$, Rayleigh-like phase function $\Phi = (1 + \cos^2(\theta - \phi)) / (3\pi)$, $f = (1 + xy)(1 + \cos^2\theta)$, zero inflow, on one column of $[0, 1]^2$ with `ndofs = [3, 3, 3]` and four angular cells (two angular refinements, as required by issue #12).
- `not_converged_warnings(caught) -> list` : the recorded `RuntimeWarning`s whose message contains `"did not converge"`.

Tests (all collective):

- `test_solve_default_ksp_settings` : without keyword arguments, `convergence_info` reports the default tolerances, `max_it = 5000` and `gmres_restart = 50`; the solve converges (`converged_reason > 0`, `converged` true) and no warning is issued.
- `test_solve_ksp_settings_from_kwargs` : with `max_it = 2, gmres_restart = 30`, the solve stops after 2 iterations with `DIVERGED_MAX_IT`, `converged` is false, and the root rank issues exactly one warning. With `rtol = 1e-4, atol = 0`, the solve stops with `CONVERGED_RTOL` and the final residual is at most $10^{-4}$ times the initial one.
- `test_solve_ksp_settings_from_petsc_options` : with the PETSc option `ksp_max_it = 3` set (and removed afterwards), the solve reports `max_it = 3`, stops after 3 iterations and is not converged.
- `test_error_high_resolution_forwards_ksp_settings` : `Error_Indicator.error_high_resolution(problem, ksp_type = "gmres", pc_type = "bjacobi", max_it = 2)` passes the solver settings on to `Problem.solve` (the returned info has `max_it = 2`, 2 iterations, not converged).
