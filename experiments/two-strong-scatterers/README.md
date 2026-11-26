`two-strong-scatterers`
================================================================================

This subdirectory contains the code and scripts used to generate the 'Two Strong Scatterers' numerical experiment results (Test Case 2, Section 4.2, Figs. 7-10, and Section SI-6) reported in

J. L. Torchinsky, S. Du, and S. N. Stechmann, Angular-spatial *hp*-adaptivity for radiative transfer with discontinuous Galerkin spectral element methods, Journal of Quantitative Spectroscopy and Radiative Transfer 348 (2026) 109687, https://doi.org/10.1016/j.jqsrt.2025.109687.

(The captions of Figs. 7 and 8 call this 'Test Case 3'.) The experiment compares three strategies that refine both in space and in angle on a problem with sharp gradients in both: a beam that is sharply peaked in angle enters through the top and left boundaries and is scattered by two optically-thick circular objects. At every trial, a random spatial-angular steering step (Eq. 16) decides whether the spatial or the angular refinements are carried out. Errors are measured against a high-resolution numerical solution. See `../README.md` for the structure shared by all experiments (solve-and-refine loop, MPI roles, stopping conditions, output file formats).

Note: There are difficulties with the stability of the numerical solve for the `hp-amr-ang-p-uni-spt` refinement strategy. This may be alleviated by changing the settings for the iterative solver, whether that be:
- the Krylov method or preconditioner used (as specified in `input.json`), or;
- the `rtol`, `atol`, `max_it`, or `gmres_restart` parameters (as specified in `solver_params` in `input.json`).

The Problem
--------------------------------------------------------------------------------

The code solves Eq. 3 on $\Omega = [0, L_x] \times [0, L_y]$ with $L_x = 3$, $L_y = 2$ (`mesh_params["Ls"]`) and $\theta \in [0, 2\pi]$, $\vec{s} = (\cos\theta, \sin\theta)$, with inflow Dirichlet boundary conditions (Eq. 4).

- Extinction coefficient (Eqs. 23a-23e): $r_1 = L_y/5 - \sqrt{(x - 9L_x/20)^2 + (y - 2L_y/5)^2}$, $\kappa_1 = 100 / (1 + e^{-30 r_1})$, $r_2 = L_y/7 - \sqrt{(x - 4L_x/5)^2 + (y - L_y/4)^2}$, $\kappa_2 = 100 / (1 + e^{-30 r_2})$, $\kappa = \kappa_1 + \kappa_2$. These are two smoothed discs, of radius $L_y/5$ centred at $(1.35, 0.8)$ and of radius $L_y/7$ centred at $(2.4, 0.5)$, with $\kappa \approx 100$ inside.
- Scattering coefficient (Eq. 23e): $\sigma = 0.7\,\kappa$.
- Scattering phase function: Henyey-Greenstein (Eq. 24) with $g = 0.8$, $\Phi(\theta, \theta') = \frac{1}{C(g)} \frac{1 - g^2}{(1 + g^2 - 2g\cos(\theta - \theta'))^{3/2}}$. The code computes $C(g) = \int_0^{2\pi} \frac{1 - g^2}{(1 + g^2 - 2g\cos\Theta)^{3/2}} d\Theta$ with `scipy.integrate.quad` at import (`Phi_norm`), so that $\Phi$ integrates to 1 over $[0, 2\pi]$ (Eq. 2).
- Forcing: $f = 0$.
- Boundary data (Eq. 25): $u^{in,bd}(x, y, \theta) = e^{-((48/(2\pi))(\theta - 7\pi/4))^2}$ if $y = L_y$ or $x = 0$, and $0$ otherwise. The direction $7\pi/4$ points down and to the right, so the beam enters through the top and left edges. The code tests `y == y_top` and `x == x_left` with exact floating-point equality; this holds at the boundary nodes of the meshes used here, because the LGL end nodes are exactly $\pm 1$ and the element edges are dyadic fractions of $L_x$ and $L_y$.

Initial Mesh, Solver and Refinement Strategies
--------------------------------------------------------------------------------

The initial mesh (`mesh.py`) has `ndofs = [3, 3, 4]` nodes per element, refined `nref_ang = 3` times in angle and `nref_spt = 2` times in space: a $4 \times 4$ grid of columns with 8 angular cells each, $16 \cdot 9 \cdot 8 \cdot 4 = 4608$ degrees of freedom, as in Section 4.2.

The linear systems are solved with PETSc `fgmres` and the `kaczmarz` preconditioner (`solver_params`), as stated in Section 4.

Each strategy has an angular part (`ang`, used with the angular-jump indicator, Eq. 14) and a spatial part (`spt`, used with the spatial-jump indicator, Eq. 13):

| Strategy key | Label in the publication | `ang`: `ref_form`, `ref_tol` | `spt`: `ref_form`, `ref_tol` |
| --- | --- | --- | --- |
| `hp-amr-spt-p-uni-ang` | *hp*-Adap. Spt., *p*-Unif. Ang. | `p`, `[0.0, 0.0]` | `hp`, `[0.0, 0.8]` |
| `hp-amr-ang-p-uni-spt` | *hp*-Adap. Ang., *p*-Unif. Spt. | `hp`, `[0.8, 0.0]` | `p`, `[0.0, 0.0]` |
| `hp-amr-ang-hp-amr-spt` | *hp*-Adap. Ang., *hp*-Adap. Spt. (Fig. 9) | `hp`, `[0.8, 0.0]` | `hp`, `[0.0, 0.8]` |

`ref_kind` is `ang` in every angular part and `spt` in every spatial part. A tolerance of `0.0` marks every element (uniform refinement); `0.8` marks elements whose indicator is at least 0.8 times the largest (Eq. 15, $\tau_{\vec{x}} = \tau_{\vec{s}} = 0.8$). With `ref_form = "hp"`, the *hp*-steering criteria (Section 3.4 and SI-3) choose *h* or *p* for each marked element.

Spatial-angular steering (Eq. 16). After computing both indicators, root computes the angular and spatial errors attempted to resolve (EA2R) as `Error_Indicator.mean_error_to_resolve()`, the indicator summed over the marked elements divided by the number of marked elements. It then draws `steer` uniformly from $[0, 1)$ with a `numpy` generator and refines spatially (`err_ind_spt.ref_by_ind()`) if `steer` $< \bar{\eta}_{\vec{x}} / (\bar{\eta}_{\vec{x}} + \bar{\eta}_{\vec{s}})$, and angularly (`err_ind_ang.ref_by_ind()`) otherwise. Before pull request #23 the driver used the sums (`error_to_resolve`) instead of the averages that Section 3.3 describes. The uniform parts of the first two strategies are therefore applied only on the trials in which the steering step picks that kind of refinement.

On output trials the driver computes the high-resolution error (`Error_Indicator.error_high_resolution`, collective) $\sqrt{\int\int (u_{hr} - u_{hp})^2 / \int\int u_{hr}^2}$, where $u_{hr}$ is solved on a mesh whose every cell has `ang_ref_offset = 2` more angular nodes and every column `spt_ref_offset = 2` more nodes in each spatial direction ("p-refining the low-resolution hp-mesh twice in both angle and space", Section 4.2.2; the initial high-resolution mesh has $16 \cdot 25 \cdot 8 \cdot 6 = 19200$ degrees of freedom, as stated there). This error is tested against `min_err` and plotted in `convergence.png`.

Section 4.2.2 reports the angular and spatial EA2R of each trial. The driver does not write them to file: `ref_strat_log.json` records only the kind of refinement chosen, and the summed indicators of the marked elements are in `err_ind_ang_jmp.json` and `err_ind_spt_jmp.json` (`error_to_resolve`).

Files
--------------------------------------------------------------------------------

### `input.json`

- `seed` : Seed for the random number generator of the steering step. If `0`, root draws `int(1e8 * (time.time() % 1))` and broadcasts it. The strategies are run in the order of `refinement_strategies`, and the $k$-th strategy ($k = 1, 2, 3$) uses `numpy.random.default_rng(seed + k)`. The seed actually used is written to that strategy's `ref_strat_log.json` (`"seed"`); to repeat the $k$-th strategy, set `seed` to the logged value minus $k$.
- `ndof_output_ratio` (`1.25`) : a trial is an output trial if its number of degrees of freedom is at least this factor times that of the previous output trial (trial 0 is always one). Only output trials compute the high-resolution error and get the full set of plots.
- `solver_params` : keyword arguments for `rt.Problem.solve`, also forwarded to the high-resolution solve.
    - `ksp_type` : The Krylov method name for the `PETSc` KSP object (default `gmres`).
    - `pc_type` : The preconditioner name for the `PETSc` PC object (default `bjacobi`).
    - `rtol`, `atol`, `divtol`, `max_it` (optional) : Tolerances and maximum number of iterations for the `PETSc` KSP object (defaults: `numpy.finfo(float64).resolution` = `1e-15`, `1e-15`, `sqrt` of the largest float, `5000`).
    - `gmres_restart` (optional) : The restart length for (F)GMRES (default: `50`).
    - `PETSc` command-line options (e.g., `-ksp_rtol`, `-ksp_max_it`, `-ksp_monitor`) override these settings. A solve that does not converge (negative `converged_reason`) raises a `RuntimeWarning`.
- `stopping_conditions` : see "Stopping Conditions" in `../README.md`. A value of `0` or less disables the condition.
    - `max_ndof` (`300000`) : Run a trial only if its mesh has at most this many degrees of freedom.
    - `max_ntrial` (`0`, disabled) : Run at most this many trials.
    - `min_err` (`1.0e-3`) : Stop once the high-resolution error of an output trial is below this value.
- `mesh_params` : Parameters for the initial mesh.
    - `Ls` : Length of the spatial domain in the $x$- and $y$-directions.
    - `pbcs` : Whether the domain is periodic in $x$ and $y$. All experiments use `[false, false]`; periodic domains are not exercised by any experiment.
    - `has_th` : Whether the mesh has an angular dimension (`true`).
    - `ndofs` : Number of nodes per element in the $x$-, $y$-, and $\theta$-directions.
    - `nref_ang` : Number of uniform angular *h*-refinements of the initial single cell ($2^{n}$ cells per column).
    - `nref_spt` : Number of uniform spatial *h*-refinements of the initial single column ($2^{n} \times 2^{n}$ columns).
- `hr_err_params` : Parameters for the high-resolution error (keyword arguments of `Error_Indicator.error_high_resolution`).
    - `ang_ref_offset` : Number of angular nodes added to every cell to obtain the high-resolution mesh.
    - `spt_ref_offset` : Number of nodes added in each spatial direction to every column.
    - Since pull request #30, every element's degree is raised by exactly this offset. Before, the offset was applied as repeated `Mesh.ref_mesh(form = "p")` sweeps, which could raise some elements more than once per sweep.
- `hp-amr-spt-p-uni-ang`, `hp-amr-ang-p-uni-spt`, `hp-amr-ang-hp-amr-spt` : Parameters for each refinement strategy.
    - `ang` : Keyword arguments of the `Error_Indicator` that computes the angular-jump indicator.
        - `ref_kind` : Refinement kind (`ang`, `spt`, or `all`).
        - `ref_form` : Refinement form (`h`, `p`, or `hp`).
        - `ref_tol` : Refinement tolerances [angle, space], each between 0 and 1 (inclusive).
    - `spt` : Keyword arguments of the `Error_Indicator` that computes the spatial-jump indicator (same keys).

### `mesh.py`

Identical to `../manufactured-solution/mesh.py`. Reads `input.json` at import and defines `mesh` (`dg.mesh.Mesh`): `Mesh(Ls, pbcs, has_th, ndofs)` refined `nref_ang` times with `ref_mesh(kind = "ang", form = "h")`, then `nref_spt` times with `ref_mesh(kind = "spt", form = "h")`. Also defines `input_dict` and `mesh_params` (`dict`).

### `problem.py`

Reads `input.json` at import (for $L_x$, $L_y$) and defines:

- `kappa(x, y)` : $\kappa_1 + \kappa_2$ of Eqs. 23a-23e. Accepts broadcastable arrays.
- `sigma(x, y)` : $0.7\,\kappa(x, y)$.
- `g` (`float`, `0.8`) : Henyey-Greenstein asymmetry parameter.
- `Phi_HG(Th)` : unnormalized Henyey-Greenstein kernel $(1 - g^2)/(1 + g^2 - 2g\cos\Theta)^{3/2}$ of the angle difference `Th`.
- `Phi_norm`, `abserr` (`float`) : $\int_0^{2\pi}$ `Phi_HG` and the `quad` error estimate (tolerances `1e-9`).
- `Phi(th, phi)` : normalized phase function `((1 - g**2) / Phi_norm) / (1 + g**2 - 2 g cos(th - phi))**1.5` (Eq. 24 with $C(g)$ = `Phi_norm`).
- `f(x, y, th)` : returns `0.` (a Python scalar for any input).
- `x_left` (`0.`), `y_top` (`Ly`) : edges with incoming radiation.
- `bcs(x, y, th)` : boundary data of Eq. 25 with angular sharpness `sth = 48.`. Returns $\exp(-((48/(2\pi))(\theta - 7\pi/4))^2)$ (same shape as `th`) when `y == y_top` or `x == x_left`, and the scalar `0` otherwise. `x` and `y` must be scalars, since the test `(y == y_top) or (x == x_left)` is a Python `or`.
- `dirac` (`[None, None, None]`), `bcs_dirac` (`[bcs, dirac]`) : boundary data in the form expected by `rt.Problem`.
- `problem` (`rt.Problem`) : `Problem(kappa, sigma, Phi, f, bcs_dirac)`.

There is no exact solution `u`.

### `refinement_strategies.py`

Reads `input.json` at import and defines `refinement_strategies` (`dict`): `{"hp-amr-spt-p-uni-ang": ..., "hp-amr-ang-p-uni-spt": ..., "hp-amr-ang-hp-amr-spt": ...}`, each value being the `input.json` entry (a `dict` with keys `ang` and `spt`). The dictionary order is the order in which the strategies are run (and hence which seed each one gets) and plotted.

### `experiment.py`

`main()` runs the solve-and-refine loop of `../README.md` for each strategy on all ranks. Before the first strategy it resolves `seed` (see `input.json`). For each strategy it creates `rng = numpy.random.default_rng(seed + k)`, and for each trial:

1. all ranks: `problem.solve(l_mesh, **solver_params)`;
2. root: writes `convergence_info.json`, `matrix_info.json`, `uh.npy`, `mesh.json`; computes `Error_Indicator(uh, **ref_strat["ang"]).error_angular_jump()` and writes `err_ind_ang_jmp.json`; computes `Error_Indicator(uh, **ref_strat["spt"]).error_spatial_jump()` and writes `err_ind_spt_jmp.json`; logs `lr_converged_reason`, `lr_res_best`;
3. on output trials: all ranks compute `error_high_resolution(problem, **solver_params, **hr_err_params)` (the `Error_Indicator` is built with `ref_strat["ang"]`, which only affects the marking stored in `err_ind_hr.json`); root writes `convergence_info_hr.json`, `matrix_info_hr.json`, `uh_hr.npy`, `mesh_hr.json`, `err_ind_hr.json` and logs `hr_converged_reason`, `hr_res_best`;
4. root: the steering step (Eq. 16) above picks `ref_kind` (`"spt"` or `"ang"`), refines with the corresponding indicator's `ref_by_ind()`, and logs `ref_kind`; the new mesh is broadcast to all ranks;
5. `min_err` is updated from the most recent high-resolution error and broadcast; the stopping conditions are checked.

After the last trial, root writes `ref_strat_log.json` (including `"seed"`). `main()` is run through `utils.run_or_abort`.

### `generate_problem_visualizations.py`

Identical to the One Weak Scatterer script. `main()` (one rank) writes `kappa.png`, `sigma.png` (`plot_xy` of $\kappa/(2\pi)$ and $\sigma/(2\pi)$ on a plotting mesh with `ndofs = [5, 5, 5]` refined 4 times with `ref_mesh(kind = "all", form = "h")`; cf. Fig. 7(a)) and `Phi.png` (`plot_Phi`; cf. Fig. 7(b)) to `out/`, skipping existing files.

`plot_Phi(Phi, file_path)` : polar plot of `Phi(0, th)` for 720 angles in $[0, 2\pi]$, i.e., $\Phi$ as a function of $\theta - \theta'$; saved at 300 dpi. Returns `None`.

### `generate_error_history.py`

`main()` (run on `N` ranks by `run.sh`). As in One Weak Scatterer: for each strategy without `tracked_vals.json`, it selects the output trials and the last trial, reads `error` from `err_ind_hr.json`, or, if that file is missing, computes `error_high_resolution` on all ranks (with `ref_strat["ang"]` as a placeholder for the marking parameters) and writes `uh_hr.npy`, `mesh_hr.json` (unless both exist), `err_ind_hr.json`, `convergence_info_hr.json` and `matrix_info_hr.json`. Root writes `tracked_vals.json` = `{ndof: high-resolution error}`.

### `generate_linear_solve_visualizations.py`

Identical to the Manufactured Solution script. `main()` plots `convergence_info.json` to `convergence_info.png` and `convergence_info_hr.json` to `convergence_info_hr.png` in every trial directory. `plot_convergence_info(convergence_info, file_path)` plots the residual history (log scale) with a line at `atol` and, if all iterations were used, at `max_it`; returns `None`.

### `generate_ang_spt_steering_plot.py`

`main()` (one rank): for each strategy whose `ref_strat_log.json` exists and `ang_spt_steering.png` does not, calls `plot_ang_spt_steering`. (Its `argparse` description, "Plots the convergence rates for each refinement strategy", is copied from `generate_convergence_plot.py`.)

`plot_ang_spt_steering(ref_strat_log, file_path)` : `ref_strat_log` is the dictionary loaded from `ref_strat_log.json`. For every numeric key (trial), plots a point at $+1$ ("Ang.") if `ref_kind == "ang"` and at $-1$ ("Spt.") otherwise, against the trial number ("Iteration"), and saves the figure at 300 dpi. The `"seed"` key is skipped. The points are collected in arrays of fixed length 1000, so a log with more than 1000 trials raises `IndexError`. Returns `None`.

### `generate_experiment_visualizations.py`

`main()` (one rank): for each output trial (and the last trial), writes `mesh.png` (cf. Fig. 9 and Figs. SI-12, SI-14, SI-16), `uh_th.png`, `uh_xy.png`, `uh_xth.png`, `uh_yth.png`, `uh_xyth.png`; `err_ind_hr.png` (`plot_error_indicator(..., kind = "all")`); `err_ind_ang_jmp.png` (`plot_refinement_thresholds(..., kind = "ang")`); `err_ind_spt_jmp.png` (`plot_refinement_thresholds(..., kind = "spt")`); and, if `mesh_hr.json` and `uh_hr.npy` exist, `mesh_hr.png` and `uh_hr_*.png` (cf. Fig. 8). Existing files are skipped.

### `generate_convergence_plot.py`

`main()` (one rank): plots every strategy's `tracked_vals.json` (high-resolution error against total degrees of freedom, log $y$-axis with limits $[10^{-3}, 0.3]$) to `out/convergence.png`, cf. Fig. 10. The plot against the number of non-zero matrix entries is not produced by any script.

### `run.sh`

Runs, from this directory, with `N` = `-n` (default 1) and `OUT_DIR = out`:

1. `mpirun -n 1 python generate_problem_visualizations.py`;
2. `mpirun -n N python experiment.py`;
3. `mpirun -n N python generate_error_history.py`;
4. `mpirun -n 1 python generate_linear_solve_visualizations.py`;
5. `mpirun -n 1 python generate_ang_spt_steering_plot.py`;
6. `mpirun -n 1 python generate_experiment_visualizations.py`;
7. `mpirun -n 1 python generate_convergence_plot.py`.

Steps 2-7 add the Open MPI option `--use-hwthread-cpus`. `EXPERIMENT_NAME` is only used in `printf` statements. The script stops at the first failed step.

Output Directory
--------------------------------------------------------------------------------

```
out
|--- kappa.png, sigma.png, Phi.png
|--- convergence.png
|--- hp-amr-spt-p-uni-ang, hp-amr-ang-p-uni-spt, hp-amr-ang-hp-amr-spt
     |--- ref_strat_log.json, tracked_vals.json, ang_spt_steering.png
     |--- 0, 1, 2, ... : mesh.json, uh.npy, convergence_info.json/.png, matrix_info.json,
                         err_ind_ang_jmp.json/.png, err_ind_spt_jmp.json/.png, and on output
                         trials err_ind_hr.json/.png, mesh_hr.json, uh_hr.npy,
                         convergence_info_hr.json/.png, matrix_info_hr.json, mesh.png,
                         uh_*.png, mesh_hr.png, uh_hr_*.png
```

The file formats are described in `../README.md`. Note that `*.npy` files are raw `float64` dumps, not NPY files.

Running the Numerical Experiment
--------------------------------------------------------------------------------

1. Create and activate the included `conda` environment by running the following commands in the root directory:

```
conda env create -f environment.yml
conda activate radiative-transfer-amr
```

2. From this directory, execute the run script using a specified number ($N$) of MPI tasks:

```
bash run.sh -n N
```

(or `chmod +x run.sh` once and then `./run.sh -n N`). The scripts read `input.json` and import `mesh.py`, `problem.py` and `refinement_strategies.py` from the current directory, so they must be run from here. On a cluster, run the script inside an allocation of at least $N$ tasks; scheduler options and the MPI launcher are site-specific. Because of the random steering step, two runs with `seed = 0` refine differently; set `seed` to a positive value to repeat a run.

To modify the experiment, edit `input.json` (mesh, solver, stopping conditions, strategies, seed), `problem.py` (coefficients, boundary data) or `refinement_strategies.py` (which strategies are run).

Results with the current code
--------------------------------------------------------------------------------

The figures below come from a run of the code at commit `f74d84c` (upstream `main` with pull requests #22-#33 merged) with the `input.json` of this directory, executed as `bash run.sh -n 24` (24 MPI ranks on one node; wall time 2.5 h; Python 3.12, NumPy 1.26.4, SciPy 1.13.1, Matplotlib 3.9.2, petsc4py 3.22.0, mpi4py 4.0.0, Open MPI 5.0.5). They were reduced to 800 pixels wide with `python experiments/readme_figures.py 800 <out/...png> figures/<name>.png`. Running `run.sh` writes the full-size versions of these and all other figures to `out/`. `input.json` has `seed = 0`, so the steering step drew a new seed; the seeds used were 16956616, 16956617 and 16956618 for the three strategies, in the order of the table below (see `seed` above to repeat them).

The problem, cf. Fig. 7: the extinction coefficient $\kappa/(2\pi)$ (`out/kappa.png`) and the Henyey-Greenstein phase function with $g = 0.8$ as a function of $\theta - \theta'$ (`out/Phi.png`).

![Extinction coefficient](figures/kappa.png)

![Scattering phase function](figures/Phi.png)

A high-resolution numerical solution, cf. Fig. 8: the angular integral $\int_0^{2\pi} u_{hr} \, d\theta$ and the integral over $x$ as a function of $(y, \theta)$ (`out/hp-amr-ang-hp-amr-spt/23/uh_hr_xy.png` and `uh_hr_yth.png`). $u_{hr}$ is the reference solution of the last trial of *hp*-Adap. Ang., *hp*-Adap. Spt., with 503949 degrees of freedom. The colour maps show values below 0 in blue. There are regions inside the two optically thick discs and in directions with almost no radiation, where the high-order solution undershoots: 5.7 % of the nodal values of this $u_{hr}$ are negative, with minimum $-0.022$. The final $u_{hr}$ of the run before the fixes (below), on a larger mesh, has 2.3 % negative values with minimum $-0.003$.

![Angular integral of the high-resolution solution](figures/uh_hr_xy.png)

![Integral over x of the high-resolution solution](figures/uh_hr_yth.png)

The final mesh of *hp*-Adap. Ang., *hp*-Adap. Spt. (`out/hp-amr-ang-hp-amr-spt/23/mesh.png`, 208393 degrees of freedom), cf. Fig. 9(a). Rectangles are spatial elements, coloured by their number of nodes in $x$; wedges are angular cells, coloured by their number of angular nodes. The small spatial elements lie around the two discs, and the angular meshes differ between spatial elements.

![Final mesh, hp-Adap. Ang., hp-Adap. Spt.](figures/mesh_hp-amr-ang-hp-amr-spt.png)

The kind of refinement chosen by the steering step (Eq. 16) at each trial of the same strategy (`out/hp-amr-ang-hp-amr-spt/ang_spt_steering.png`): 12 angular and 12 spatial refinements.

![Spatial-angular steering](figures/ang_spt_steering_hp-amr-ang-hp-amr-spt.png)

The high-resolution error against the total number of degrees of freedom (`out/convergence.png`), cf. the left panel of Fig. 10:

![Convergence](figures/convergence.png)

| Strategy | Trials | Final degrees of freedom | Final high-resolution error | Before the fixes |
| --- | --- | --- | --- | --- |
| *hp*-Adap. Spt., *p*-Unif. Ang. | 13 | 258272 | 7.24e-3 | 7.49e-3 (288432, 26 trials) |
| *hp*-Adap. Ang., *p*-Unif. Spt. | 31 | 291150 | 4.87e-3 | 2.35e-2 (286399, 27 trials) |
| *hp*-Adap. Ang., *hp*-Adap. Spt. | 24 | 208393 | 7.81e-3 | 3.64e-3 (269608, 54 trials) |

All strategies stopped at `max_ndof` $= 300000$; none reached `min_err` $= 10^{-3}$.

**Linear solver.** As the note at the top of this README warns, the solves of *hp*-Adap. Ang., *p*-Unif. Spt. do not converge on larger meshes. From trial 9 onwards, 22 of its low- and high-resolution solves stopped at `max_it` $= 5000$ iterations with relative residuals up to $2.6 \times 10^{-2}$. Each such solve raised the `RuntimeWarning` added in pull request #28; the converged reasons are in `ref_strat_log.json`. Every plotted point of that strategy from 55616 degrees of freedom onwards involves such a solve, so these errors include an unknown solver error. This includes the rise from 0.020 to 0.039 between 98000 and 176000 degrees of freedom and the final value. The other two strategies each had one high-resolution solve that did not converge (trial 10 of *hp*-Adap. Spt., *p*-Unif. Ang. and trial 16 of *hp*-Adap. Ang., *hp*-Adap. Spt.). The run before the fixes had 40 such solves in this experiment (12, 24 and 4, in the order of the table), and none of them raised a warning.

**Comparison with the publication.** Section 4.2.2 and Fig. 10 report that *hp*-Adap. Spt., *p*-Unif. Ang. is the least efficient strategy and *hp*-Adap. Ang., *hp*-Adap. Spt. the best, especially beyond about 75000 degrees of freedom. In this run *hp*-Adap. Ang., *hp*-Adap. Spt. is more accurate than *hp*-Adap. Spt., *p*-Unif. Ang. at comparable mesh sizes (for example 0.012 at 155000 against 0.016 at 177000). *hp*-Adap. Ang., *p*-Unif. Spt. has the smallest error between 30000 and about 100000 degrees of freedom. However, only its points up to 42581 degrees of freedom come from converged solves (see above). The publication's final *hp*-Adap. Ang., *hp*-Adap. Spt. mesh has 248160 degrees of freedom and a high-resolution counterpart of over one million; here they are 208393 and 503949.

**Comparison with the code before the fixes.** "Before the fixes" is a run of the same scripts at upstream `main` (commit `8b7cc7f`, with only the module-name fix of pull request #22 applied) on the same machine type and number of ranks (wall time 4.0 h). The steering step is random and both runs drew their seeds from the clock, so the sequence of spatial and angular refinements differs between the runs; differences in individual error histories are expected even without code changes. Several fixes change the computation in this experiment:

- the steering step now uses averaged rather than summed indicators (#23, issues #7-#9), which changes how often each kind of refinement is chosen;
- the spatial and angular *hp*-steering formulas (#23);
- the angular upwind flux (#24, issue #10);
- the high-resolution reference (#30, issue #16).

The fixed *hp*-Adap. Ang., *hp*-Adap. Spt. run needed 24 trials instead of 54 to reach about 200000 degrees of freedom. Its final error, 7.8e-3 at 208393, compares with 7.1e-3 at 182000 and 4.8e-3 at 227000 before the fixes.
