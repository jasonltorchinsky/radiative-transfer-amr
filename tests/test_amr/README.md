`test_amr`
================================================================================

This subdirectory contains regression tests for the code in `src/amr`, i.e., the `Error_Indicator` class of `amr.error_indicator`: the reference errors (`error_analytic`, `error_high_resolution`), the angular-jump indicator, the *hp*-steering criteria (`cell_hp_steer`, `col_hp_steer`) and the average used by the spatial-angular steering criterion (`mean_error_to_resolve`). In the solve/refinement loop of the experiment drivers these functions run on the root rank after `rt.Problem.solve` and before `Error_Indicator.ref_by_ind` refines the mesh.

All tests here build small meshes with `dg.mesh.Mesh` and interpolate known functions onto them with `dg.projection.Projection(mesh, u)`, so no linear solve is needed, except `test_error_high_resolution.py`, which replaces the solver by a stub. They run serially; under `--with-mpi` each rank runs them independently.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `test_amr` subdirectory are:

- `test_error_analytic.py` : Issue #17 (PR #31), `Error_Indicator.error_analytic`.
- `test_error_high_resolution.py` : Issue #16 (PR #30), the high-resolution reference mesh of `error_high_resolution`.
- `test_hp_steering.py` : Issues #7, #8, #9 (PR #23), the *hp*-steering and spatial-angular steering formulas.
- `test_imports.py` : Issue #6 (PR #22), the name of the `amr.error_indicator` package.

`test_error_analytic.py`
--------------------------------------------------------------------------------

Regression tests for issue #17. `Error_Indicator.error_analytic(anl_sol)` computes the relative $L^2$ error

$e = \left( \int (u - u_{hp})^2 \right)^{1/2} / \left( \int u^2 \right)^{1/2}$

per cell, per column and on the whole mesh. Before the fix it integrated with a fixed rule of $n + 3$ nodes per direction, which under-resolves narrow analytic solutions on coarse elements, and it stored $\int u^2$ from the first call and reused it in later calls.

Module-level helpers (not tests):

- `Ls = [3., 2.]` : domain size $[L_x, L_y]$ used by `u` and `coarse_mesh`.
- `u(x, y, th)` : the manufactured solution of `experiments/manufactured-solution`, $u = e^{-((x - L_x/3)/L_x)^2} \, e^{-4 (L_y - y)/L_y} \, e^{-(96 (\theta - 7\pi/5) / (2\pi))^2}$. Its angular profile has a standard deviation of about $0.046$ rad. Accepts broadcastable `numpy` arrays.
- `coarse_mesh() -> Mesh` : one column on $[0, 3] \times [0, 2]$ with `ndofs = [5, 5, 3]`, angularly *h*-refined three times (eight cells of width $\pi/4$), as in the initial angular mesh of the manufactured-solution experiment.
- `reference_error(proj, nnodes_xy, nnodes_th) -> float` : the relative $L^2$ error of `proj` against `u` with a fixed fine rule (`nnodes_xy` Gauss-Lobatto nodes in $x$ and $y$, `nnodes_th` Gauss nodes in $\theta$ on every element). It evaluates $u_{hp}$ at the fine nodes from the Lagrange basis (`lag_eval`) and `cell.vals` with index order `[i, j, a]` ($x$, $y$, $\theta$), and computes $\int u^2$ with `scipy.integrate.nquad`.

Tests:

- `test_error_analytic_zero_projection` : for $u_{hp} = 0$ (`Projection(coarse_mesh())`) the relative error is exactly 1; `err_ind.error` must be within $10^{-2}$ of 1. With the old fixed rule of $n + 3 = 6$ angular nodes per $\pi/4$-wide cell the narrow peak was under-resolved and the value was wrong.
- `test_error_analytic_matches_fine_quadrature` : for the interpolant of `u` on `coarse_mesh()`, `error_analytic` must agree with `reference_error(proj, 16, 40)` to a relative tolerance of $10^{-2}$.
- `test_error_analytic_normalization_per_call` : calls `error_analytic` on the zero projection of a $3 \times 3 \times 3$ one-element mesh, first with $v = (1 + x + y)(1 + \theta)$ and then with $w = 2v$. Both are integrated exactly, so each call must return an error of 1 (to $10^{-6}$). With a normalization cached from the first call, the second call would return 2.

`test_error_high_resolution.py`
--------------------------------------------------------------------------------

Regression test for issue #16. `error_high_resolution(err_ind, problem, ang_ref_offset = ..., spt_ref_offset = ...)` solves the problem again on a reference mesh. That mesh must be the current mesh with the number of nodes of every leaf column raised by exactly `spt_ref_offset` in $x$ and in $y$, and of every leaf cell by exactly `ang_ref_offset` in $\theta$, with the same column and cell keys. Before the fix the reference mesh was built with repeated `Mesh.ref_mesh(form = "p")` sweeps, whose even-degree cascade raised some elements more than once.

Module-level helpers:

- `_MeshCaptured` : exception raised by the stub solver.
- `_CaptureProblem()` : stands in for `rt.Problem`. The constructor takes no arguments and sets `self.mesh = None`. `solve(mesh, **kwargs)` stores `mesh` in `self.mesh` and raises `_MeshCaptured`, so the test can inspect the reference mesh without solving.
- `_non_uniform_mesh() -> Mesh` : $[0, 2] \times [0, 1]$, `ndofs = [3, 3, 3]`, refined twice in angle and once in space, then one cell *p*-refined ($n_\theta = 4$) and one column *p*-refined ($n_x = n_y = 4$), so that degrees differ between elements.

Test:

- `test_high_resolution_mesh_offset_is_uniform[ang_ref_offset, spt_ref_offset]`, parametrized over `(1, 0)`, `(0, 1)`, `(2, 2)` : calls `error_high_resolution` with a `types.SimpleNamespace` in place of the `Error_Indicator` (only `err_ind.proj.mesh` is read before `solve`), and checks that the captured mesh has the same column and cell keys and that every `ndofs` is raised by exactly the offset. `error_high_resolution` broadcasts the mesh from the root rank before calling `solve`, so the test is collective under MPI.

`test_hp_steering.py`
--------------------------------------------------------------------------------

Regression tests for the steering formulas.

Module-level helpers:

- `make_mesh(n) -> Mesh` : one column $[0, 2]^2$ (area 4) with `ndofs = [n, n, n]`, angularly *h*-refined twice (four cells, one per quadrant).
- `expected_form(lp, n) -> str` : `"p"` if $l_p - 1/2 \geq n$, otherwise `"h"`, the decision rule that follows Eqs. 18b and SI-6b.

Tests:

- `test_cell_hp_steer_follows_eq_18b[nth]`, `nth` in 3, 4, 5 (issue #8) : angular *hp*-steering, Eq. 18b, $l_p^\theta = \log\left( (2 n_\theta - 1) / (2 |\gamma_f|^2) \right) / \left( 2 \log(n_\theta - 1) \right)$. The test sets $u = c P_{n_\theta - 1}(\hat{\theta})$ on the first cell, so that the spatial integral $g(\theta) = 4 c P_{n_\theta-1}$ has top Legendre coefficient $\gamma_f = 4c$. For seven values of $\gamma_f$ between $0.3$ and $4$ times the threshold value (where $l_p - 1/2 = n_\theta$) it checks that `Error_Indicator.cell_hp_steer(0, cell_key)` returns `expected_form`. Some of these values gave the opposite decision with the factor $2/(2 n_\theta - 1)$ that the code applied before the fix (the squared $L^2$ norm of the top mode instead of $|\gamma_f|^2$).
- `test_col_hp_steer_follows_si_eq_6b[n, shape]`, `n` in 3, 4, 5 and `shape` in `"sum"`, `"product"` (issue #9) : spatial *hp*-steering, SI Eq. 6b. With $u$ independent of $\theta$, $g = 2\pi u$; the test chooses $g = c (P_p(\hat{x}) + P_p(\hat{y}))$ (`"sum"`, $(\gamma^x_f)^2 = (\gamma^y_f)^2 = 2 c^2$) or $g = c P_p(\hat{x}) P_p(\hat{y})$ (`"product"`, $(\gamma^x_f)^2 = (\gamma^y_f)^2 = 2 c^2 / (2p + 1)$), with $p = n - 1$. For seven values of $c$ around the threshold it checks `Error_Indicator.col_hp_steer(0)`. The `"product"` case checks the normalization of the top index: the $n$-point Gauss-Lobatto rule is not exact for $P_{n-1}^2$, so the code normalizes the top coefficient by $(n-1)/2$ instead of $(2n-1)/2$ to obtain the exact coefficient.
- `test_mean_error_to_resolve_is_average_over_marked_elements` (issue #7) : builds two indicators by hand (`Error_Indicator(None, ...)` with `cols` filled with `Error_Indicator_Column` and `Error_Indicator_Cell` objects): a spatial one with 4 columns of which 2 are marked (errors 1.0 and 0.9), and an angular one with 32 marked cells of error about 0.1. It checks that `mean_error_to_resolve()` returns the averages ($0.95$ and the mean cell error), that the probability of spatial refinement of Eq. 16, $p_x = \bar\eta_x / (\bar\eta_x + \bar\eta_s)$, is above $0.85$, and that the ratio of the sums would have given $p_x < 0.5$.
- `test_mean_error_to_resolve_after_angular_jump` : for a smooth $u = e^{\cos\theta}(1 + xy)$ on a $2 \times 2$-column mesh and `ref_kind` in `"ang"`, `"spt"`, `"all"`, runs `error_angular_jump()` and checks that `mean_error_to_resolve()` equals the mean error over the marked columns and cells.

The `Error_Indicator` methods tested here are documented in `src/amr/error_indicator/README.md`.

`test_imports.py`
--------------------------------------------------------------------------------

Regression test for issue #6. The package directory was named `src/amr/Error_Indicator` while every import uses `amr.error_indicator`, which fails on case-sensitive file systems.

- `test_error_indicator_package_name_matches_imports` : the directory of `amr` contains a subdirectory `error_indicator` and none named `Error_Indicator`.
- `test_error_indicator_imports` : `importlib.import_module("amr.error_indicator")` succeeds and exposes `Error_Indicator`, and `amr.error_indicator.ref_by_ind` imports.
