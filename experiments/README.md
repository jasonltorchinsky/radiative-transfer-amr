`experiments`
================================================================================

This directory contains the drivers for the numerical experiments reported in

J. L. Torchinsky, S. Du, and S. N. Stechmann, Angular-spatial *hp*-adaptivity for radiative transfer with discontinuous Galerkin spectral element methods, Journal of Quantitative Spectroscopy and Radiative Transfer 348 (2026) 109687, https://doi.org/10.1016/j.jqsrt.2025.109687,

and one test of the solver that requires MPI (`test-rt`). Equation numbers refer to the published article; numbers prefixed with 'SI' refer to its Supplemental Information.

Each subdirectory is self-contained: it defines a problem (extinction coefficient $\kappa$, scattering coefficient $\sigma$, scattering phase function $\Phi$, forcing $f$ and inflow boundary data), an initial mesh, and (for the three paper experiments) a set of refinement strategies. It then runs the solve-and-refine loop on top of the packages in `src`, and post-processes the results into figures.

How It Fits Into the Whole
--------------------------------------------------------------------------------

The experiments are the top of the call graph. Nothing in `src` imports from `experiments`. The only other code that reads these files is `tests/test_experiments/test_experiment_scripts.py`, which runs each `run.sh` with stand-in `mpirun` and `python` executables and reads the `keep_going` expressions of each `experiment.py` with `ast`.

The experiment files use the following parts of `src`:

- `mesh.py` builds the initial `dg.mesh.Mesh` and refines it uniformly with `Mesh.ref_mesh`.
- `problem.py` defines $\kappa$, $\sigma$, $\Phi$, $f$ and the boundary data, and wraps them in an `rt.Problem`.
- `experiment.py` calls `rt.Problem.solve` (collective over all MPI ranks), computes error indicators with `amr.error_indicator.Error_Indicator` (`error_angular_jump`, `error_spatial_jump`, `error_analytic` on the root rank; `error_high_resolution`, which solves a second problem and is collective), refines with `Error_Indicator.ref_by_ind`, and broadcasts the new mesh with `mpi4py`. It wraps `main()` in `utils.run_or_abort`, so that an exception on one rank aborts the whole MPI job.
- `generate_*.py` read the saved results back with `dg.projection.from_file` and `amr.error_indicator.from_file` and plot them with `tools.dg.mesh`, `tools.dg.projection`, `tools.amr` and `matplotlib`.

The loop in each paper experiment's `experiment.py` follows the refinement algorithm of Section 3.2. For each refinement strategy, starting from a deep copy of the initial mesh, each trial (iteration) does the following. "Root" is rank `consts.COMM_ROOT` = 0.

1. All ranks call `problem.solve(mesh, **solver_params)`. The solution `uh` (a `dg.projection.Projection`) exists only on root; it is `None` on the other ranks.
2. Root writes the solution, the mesh and the linear-solve information to `out/<strategy>/<trial>/`, then computes the jump error indicators (Eqs. 13-14). These mark the elements whose indicator is at least the refinement tolerance times the maximum indicator (Eq. 15), and choose *h*- or *p*-refinement for each marked element (Section 3.4).
3. On output trials (the first trial, and each later trial whose number of degrees of freedom has grown by at least the factor `ndof_output_ratio` since the previous output trial), the driver also computes reference errors: the analytic error (SI Eq. 8; Manufactured Solution only, root only) and the high-resolution error (all ranks).
4. Root refines the mesh with `ref_by_ind`. In Two Strong Scatterers it first picks spatial or angular refinement at random with the probability of Eq. 16. The new mesh is broadcast to all ranks.
5. The stopping conditions are evaluated for the next trial, and the loop repeats.

Uniform refinement is expressed in the same framework: a refinement tolerance of `0.0` marks every element, because every indicator is at least $0 \times$ the maximum. (The code marks an element when its indicator is `>=` the threshold of Eq. 15; step 4 of Section 3.2 says "exceeds".)

Common Files
--------------------------------------------------------------------------------

The three paper experiments share the same file layout. `test-rt` has only `input.json`, `mesh.py`, `problem.py`, `experiment.py` and `run.sh`.

| File | Role |
| --- | --- |
| `input.json` | All parameters: initial mesh, linear solver, stopping conditions, high-resolution offsets, refinement strategies. |
| `mesh.py` | Builds the initial mesh `mesh` from `input.json`. Identical in all four directories. |
| `problem.py` | Defines the problem and the `rt.Problem` object `problem`. |
| `refinement_strategies.py` | Builds the dictionary `refinement_strategies` (strategy name -> parameters) from `input.json`. |
| `experiment.py` | The solve-and-refine driver (MPI). |
| `generate_problem_visualizations.py` | Plots $\kappa$, $\sigma$, $\Phi$ (and the exact solution for the Manufactured Solution). |
| `generate_error_history.py` | Collects the reference error of each output trial into `tracked_vals.json`, computing missing values. |
| `generate_linear_solve_visualizations.py` | Plots the Krylov residual history of each linear solve. Identical in all three experiments. |
| `generate_experiment_visualizations.py` | Plots meshes, solutions and error indicators for each output trial. |
| `generate_convergence_plot.py` | Plots the error against the number of degrees of freedom for all strategies. |
| `generate_ang_spt_steering_plot.py` | (Two Strong Scatterers only) Plots whether each trial refined in space or in angle. |
| `run.sh` | Runs the steps above in order. |

### Working directory and imports

`mesh.py`, `problem.py` and `refinement_strategies.py` open `input.json` from the *current working directory* when they are imported, and `experiment.py` and the `generate_*.py` scripts import them as top-level modules (`from mesh import mesh`). The scripts that are run directly (`experiment.py`, `generate_*.py`) append `../../src` (relative to the script file) to `sys.path`. Therefore every script must be run from inside its experiment directory, which is what `run.sh` does. `mesh.py` and `problem.py` do not add `src` to `sys.path` themselves, so they can only be imported after `src` is on the path.

### Calling conventions in `problem.py`

`rt.Problem` stores the callables and the assembly routines in `src/rt` call them as follows:

- `kappa(x, y)` and `sigma(x, y)` are called with arrays `x` of shape `[nx, 1]` and `y` of shape `[1, ny]` (the LGL nodes of a column) and must broadcast to `[nx, ny]`. They take no angular argument, although the type hints in `rt.Problem` show three.
- `Phi(th, phi)` is called with two scalar angles. It must satisfy $\int_0^{2\pi} \Phi(\theta, \theta')\,d\theta' = 1$ (Eq. 2).
- `f(x, y, th)` is called with scalars at every node.
- `bcs(x, y, th)` (the first entry of `bcs_dirac`) is called with scalars at the nodes of the inflow boundary. The second entry of `bcs_dirac`, `dirac = [None, None, None]`, means that no Dirac-delta boundary source is used.
- The exact solution `u(x, y, th)` (Manufactured Solution) is also passed to `dg.projection.Projection` and to `Error_Indicator.error_analytic`, which call it with broadcastable arrays.

### Command-line arguments

Every script accepts `--o OUT_DIR` (default `out`; `figs` for `test-rt/experiment.py`), the output directory, relative to the experiment directory. `run.sh` always passes `--o out`.

Stopping Conditions
--------------------------------------------------------------------------------

`stopping_conditions` in `input.json` controls when the loop for one refinement strategy ends. A value of `0` or less disables that condition (it is replaced by `consts.INF`). Trials are numbered from `0`. Trial `trial` runs only if all three of the following hold:

- `max_ndof` : the mesh for that trial has at most `max_ndof` degrees of freedom;
- `max_ntrial` : `trial < max_ntrial`, i.e., at most `max_ntrial` trials are run (this changed from `<=` in pull request #32; before that, `max_ntrial = N > 0` ran `N + 1` trials);
- `min_err` : the smallest reference error recorded so far is at least `min_err`. The reference error is the analytic error (Manufactured Solution) or the high-resolution error (One Weak Scatterer, Two Strong Scatterers).

The reference error is recomputed only on output trials. On the other trials, `min_err` is updated with the value from the most recent output trial, so the `min_err` condition can end the loop only after an output trial.

Output Layout
--------------------------------------------------------------------------------

Running `run.sh` in a paper experiment produces:

```
out
|--- kappa.png, sigma.png, Phi.png         : problem visualizations
|--- u_th.png, u_xy.png, ...               : exact solution (Manufactured Solution only)
|--- convergence.png                       : error vs. degrees of freedom, all strategies
|--- <strategy>                            : one directory per key of refinement_strategies
     |--- ref_strat_log.json               : per-trial log of the linear solves
     |--- tracked_vals.json                : {ndof : reference error} for the output trials
     |--- ang_spt_steering.png             : (Two Strong Scatterers only)
     |--- 0, 1, 2, ...                     : one directory per trial
          |--- mesh.json, uh.npy           : mesh and numerical solution
          |--- convergence_info.json/.png  : linear solve information and residual plot
          |--- matrix_info.json            : PETSc information on the system matrix
          |--- err_ind_ang_jmp.json/.png   : angular-jump error indicator
          |--- err_ind_spt_jmp.json/.png   : spatial-jump error indicator (Two Strong Scatterers)
          |--- (output trials only)
          |--- err_ind_anl.json/.png       : analytic error indicator (Manufactured Solution)
          |--- err_ind_hr.json/.png        : high-resolution error indicator
          |--- mesh_hr.json, uh_hr.npy     : high-resolution mesh and solution
          |--- convergence_info_hr.json/.png, matrix_info_hr.json
          |--- mesh.png, uh_*.png, mesh_hr.png, uh_hr_*.png : plots
```

File formats:

- `uh.npy`, `uh_hr.npy` : despite the extension, these are *raw* little-endian `float64` dumps written by `ndarray.tofile` (no NPY header). `np.load` fails on them. Read them with `dg.projection.from_file(mesh_file_path, proj_file_path)`, or with `np.fromfile(path, dtype = "<f8")` together with `mesh.json`. Columns are stored in sorted column-key order, cells in sorted cell-key order within a column, and within an element the index is $n_\theta n_y i + n_\theta j + a$ ($\theta$ fastest, $x$ slowest). SI-9 Eq. (22a) instead defines the local index as $n_x n_y a + n_x j + i$.
- `mesh.json`, `mesh_hr.json` : written by `Mesh.to_file`, with keys `Ls`, `pbcs`, `has_th` and `cols` (per column: `pos`, `idx`, `lv`, `key`, `is_lf`, `ndofs`, `nhbr_keys`, `cells`).
- `err_ind_*.json` : written by `Error_Indicator.to_file`, with keys `ref_kind`, `ref_form`, `ang_ref_tol`, `spt_ref_tol`, `error` (the mesh-level error), `error_to_resolve` (the sum of the indicators of the marked elements), `col_max_error`, `cell_max_error` and `cols` (per column: `do_ref`, `ref_form`, `error`, `cells` with the same keys per cell).
- `convergence_info*.json` : the dictionary returned by `rt.Problem.solve`: `converged_reason` (PETSc `KSPConvergedReason`; negative means the solve did not converge), `converged`, `convergence_history` (residual norm per iteration), `iteration_number`, `res_f` (final residual), `res_best` (smallest residual), `rtol`, `atol`, `divtol`, `max_it`, `gmres_restart`.
- `matrix_info*.json` : PETSc `Mat.getInfo()` of the interior system matrix, summed over ranks (`nz_used` is the number of stored non-zero entries; the publication uses the number of non-zero entries of the system matrix as a cost measure, but no script here plots it), plus `extreme_singular_values` = `[emax, emin]` estimated by the Krylov solver.
- `ref_strat_log.json` : `{trial : {"lr_converged_reason", "lr_res_best", ["hr_converged_reason", "hr_res_best"]}}`. Two Strong Scatterers also stores `"seed"` at the top level and `"ref_kind"` (`"ang"` or `"spt"`) for each trial. JSON turns the integer trial keys into strings.
- `tracked_vals.json` : `{ndof : error}` (keys are strings in JSON).

Running the Experiments
--------------------------------------------------------------------------------

1. Create and activate the `conda` environment from the repository root:

```
conda env create -f environment.yml
conda activate radiative-transfer-amr
```

2. Change into the experiment directory and run the script with $N$ MPI ranks:

```
cd experiments/manufactured-solution
bash run.sh -n N
```

`-n` defaults to `1`. `run.sh` stops at the first failed step (`set -euo pipefail`). It calls `mpirun` directly, with the Open MPI option `--use-hwthread-cpus` for most steps. Only `experiment.py` and `generate_error_history.py` (and `test-rt/experiment.py`) use $N$ ranks; the plotting steps use one rank.

On a cluster, run `run.sh` inside an allocation with at least $N$ tasks (for example from a batch script). Partition names, module loads, and whether `mpirun` or `srun` should launch MPI programs are site-specific. The three paper experiments solve many large linear systems, including the high-resolution solves on output trials, so they need far more time and memory than `test-rt` (see the note on memory in `manufactured-solution/README.md`).

Reruns: `experiment.py` always recomputes and overwrites the trial directories. The `generate_*.py` scripts skip any figure that already exists, and `generate_error_history.py` skips a strategy whose `tracked_vals.json` already exists. Delete those files to regenerate them.

README Figures
--------------------------------------------------------------------------------

The 'Results with the current code' section of each experiment's README shows figures from a rerun of the current code. They are stored in `<experiment>/figures/`, which `.gitignore` exempts from its `*.png` rule. The figures were copied from the run's `out/` directory at a reduced size with `readme_figures.py`. The README of each experiment states the commit, command and number of MPI ranks of the run. The full-size figures are not in the repository; running `run.sh` regenerates them.

### `readme_figures.py`

`python experiments/readme_figures.py WIDTH SRC_PNG DST_PNG [SRC_PNG DST_PNG ...]` (from the repository root, one rank). It needs only Pillow, which is in `environment.yml`.

- `shrink(src, dst, width)` : opens `src`, converts it to RGB, resizes it to `width` pixels wide with Lanczos resampling if it is wider (keeping the aspect ratio), quantizes it to a 256-colour palette (median cut), creates the directory of `dst` if needed and saves it with PNG optimization. Returns `None`.
- `main()` : reads `WIDTH` and the source/destination pairs from `sys.argv`, exits with the usage text if no pairs are given or an argument is unpaired, and calls `shrink` for each pair, printing the size of each written file.

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `experiments` subdirectory is

```
experiments
|
|--- manufactured-solution : Test Case 1 'Manufactured Solution' (Section 4.1): known solution, four angular refinement strategies.
|--- one-weak-scatterer : Test Case 'One Weak Scatterer' (Section SI-7): one optically-thin scatterer, four angular refinement strategies.
|--- test-rt : A single solve of the Manufactured Solution problem on a fixed mesh, plotting the system matrices and the solution; requires MPI.
|--- two-strong-scatterers : Test Case 'Two Strong Scatterers' (Section 4.2): two optically-thick scatterers, three combined spatial-angular refinement strategies.
|--- readme_figures.py : Copies figures into an experiment's figures/ directory at a reduced size, for the README files.
```
