`test_experiments`
================================================================================

This subdirectory contains regression tests for the scripts in `experiments`: the solve/refinement drivers `experiments/<experiment>/experiment.py` and the `run.sh` scripts that run each experiment with `mpirun`. The tests do not import the drivers and do not solve anything. They parse `experiment.py` with `ast`, and they run each `run.sh` with fake `mpirun` and `python` executables.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `test_experiments` subdirectory are:

- `test_experiment_scripts.py` : Issue #18 (PR #32), the `max_ntrial` stopping condition and the error handling of `run.sh`.

`test_experiment_scripts.py`
--------------------------------------------------------------------------------

Regression tests for issue #18:

- the drivers' `max_ntrial` stopping condition must allow exactly `max_ntrial` trials (before the fix the condition used `<` before the loop and `<=` inside it, which gave one extra trial);
- each `run.sh` must stop at the first failed step, return a non-zero exit status, and not call `mpirun` with an empty process count when `-n` is not given.

Module-level constants and helpers:

- `EXPERIMENTS_DIR` : `<repository>/experiments`, found from the location of the test file.
- `DRIVER_EXPERIMENTS` : `["manufactured-solution", "one-weak-scatterer", "two-strong-scatterers"]`, the experiments with an adaptive `experiment.py` driver.
- `RUN_SH_EXPERIMENTS` : `DRIVER_EXPERIMENTS` plus `"test-rt"`, the experiments with a `run.sh`.
- `FAKE_EXE` : a Bash script used as both `mpirun` and `python`. It appends `"<name> <arguments>"` to the file `$FAKE_LOG` and exits with status `$FAKE_STATUS`.
- `keep_going_exprs(experiment) -> list` : parses `experiments/<experiment>/experiment.py` and returns, in source order, the compiled right-hand sides of the annotated assignments `keep_going: bool = ...` (the one before the refinement loop and the one at the end of each trial).
- `FakeMesh` : stands in for `l_mesh`; `get_ndof()` returns 1.
- `run_script(experiment, args, status) -> (CompletedProcess, list)` : copies `run.sh` into a `tempfile.TemporaryDirectory()`, puts the fake `mpirun` and `python` first on `PATH`, runs `bash run.sh <args>` there (60 s timeout), and returns the completed process and the logged calls. The temporary directory (including the `out` directory that `run.sh` creates) is deleted afterwards.

Tests:

- `test_max_ntrial_counts_trials[max_ntrial, experiment]`, `max_ntrial` in 1, 2, 5 : evaluates the two `keep_going` expressions of the driver in a namespace with `l_mesh = FakeMesh()`, `min_err = inf` and `stopping_conditions = {"max_ndof": inf, "max_ntrial": max_ntrial, "min_err": 0.}`, in a loop with the same structure as the driver (incrementing `trial`), and asserts that the loop runs exactly `max_ntrial` times.
- `test_run_sh_stops_on_failure[experiment]` : with every fake command failing (`status = 1`) and `-n 2`, `run.sh` exits non-zero, does not print `"experiment complete"`, and makes exactly one call (it stops after the first failed step).
- `test_run_sh_default_nprocs[experiment]` : with every fake command succeeding and no arguments, `run.sh` exits 0, prints `"experiment complete"`, and every logged call is `mpirun ... -n <k> ...` with an integer $k \geq 1$.

These tests need `bash` but not a real MPI installation, and they write only to the temporary directory.
