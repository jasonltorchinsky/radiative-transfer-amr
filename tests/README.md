`tests`
================================================================================

This subdirectory contains the unit and regression tests for the code in `src` and for the experiment scripts in `experiments`. The tests use the `pytest` framework and, optionally, the `pytest-mpi` plugin. Both are part of the `conda` environment in `environment.yml`.

The tests are not imported by any package code. They exercise the same entry points that the experiment drivers use: the *hp*-mesh (`dg.mesh`), projections onto it (`dg.projection`), the interior/boundary masks (`dg.matrix`), the DG assembly and linear solve (`rt.Problem.solve`), the error indicators and *hp*-steering (`amr.error_indicator`), the MPI abort wrapper (`utils.run_or_abort`), and the `run.sh`/`experiment.py` scripts.

Two kinds of tests are present.

- The original unit tests (`test_dg/test_mesh/test_Mesh.py`, `test_Column.py`, `test_Cell.py`, `test_dg/test_projection/test_Projection.py`) build small objects, check equality and file round trips, and write text and plot files. They check that the code runs and that `to_file`/`from_file` are inverses; they do not check numerical values.
- The regression tests added with the fix pull requests #22-#33 each start with a module docstring that names the GitHub issue they cover. They check numerical properties (constant preservation, exact quadrature, formulas of the paper) or error behavior.

Configuration: `conftest.py`
--------------------------------------------------------------------------------

`conftest.py` is loaded by `pytest` before any test in this directory tree is collected. It

- appends the absolute path of `src` (`<repository>/src`) to `sys.path` if it is not already there, so that the tests can import the packages as top-level modules (`import consts`, `from dg.mesh import Mesh`, `from rt import Problem`, ...), in the same way the experiment drivers do;
- imports `pytest`, `pytest_mpi` and `mpi4py.MPI`. Importing `mpi4py.MPI` initializes MPI when the test session starts. Importing `pytest_mpi` makes `pytest` fail early if the plugin is missing.

It defines no active fixtures. The commented-out fixture `mpi_tmp_path_fixed` is a disabled attempt to give all MPI ranks one shared temporary directory; it is not used by any test.

There is no `pytest.ini`, `setup.cfg` or `pyproject.toml`, so `pytest` uses its defaults: files named `test_*.py`, functions named `test_*`, and the `prepend` import mode. The test directories have no `__init__.py`, so every test module is imported under its bare file name, and two test files with the same name in different subdirectories would clash. Keep test file names unique (for this reason the Krylov-settings test of issue #14 is named `test_rt/test_solve_krylov.py` rather than a second `test_solve.py`).

Running the Tests
--------------------------------------------------------------------------------

From the `tests` directory (or from the repository root with `pytest tests`), run all tests serially with

```python -m pytest```

A single file, or a single test, is selected in the usual way, e.g.,

```python -m pytest test_rt/test_get_Eth.py::test_get_Eth_constant_trace```

To run the tests under MPI with the `pytest-mpi` plugin, start one `pytest` process per rank and pass `--with-mpi`:

```mpirun -n 2 python -m pytest --with-mpi```

Every rank runs the whole test session. Tests that call collective functions (`Problem.solve`, `Error_Indicator.error_high_resolution`, `get_intr_mask_mpi`, `get_intr_mask_seq` with its barrier) then run in parallel across the ranks; all other tests run once on each rank. Tests that read the solution check `uh` only on the root rank (`consts.COMM_ROOT` = 0), since `Problem.solve` returns `uh = None` on the other ranks.

`pytest-mpi` markers used here:

- `@pytest.mark.mpi_skip`: the test is skipped when `--with-mpi` is given. `test_utils/test_run_or_abort.py` uses it because it launches its own `mpirun`.

No test is marked `@pytest.mark.mpi` (tests that only run with `--with-mpi`), so a serial run executes every test.

Output Files
--------------------------------------------------------------------------------

The tests do not write into the repository. Tests that write files use `pytest`'s `tmp_path` fixture, a fresh directory per test under `pytest`'s base temporary directory (by default `<system temp>/pytest-of-<user>/pytest-<N>/<test name>0/`; `pytest` keeps the directories of the last three sessions). Each MPI rank is a separate `pytest` process with its own base temporary directory. Do not pass one `--basetemp` directory to several ranks, since `pytest` empties that directory at start-up.

| Test | Files written to `tmp_path` |
| --- | --- |
| `test_dg/test_mesh/test_Mesh.py` | `mesh_str.txt`, `mesh.json`, `mesh_0.png` ... `mesh_9.png`, `nhbrs_<col>_NONE.png` and `nhbrs_<col>_<cell>.png` for every column and cell |
| `test_dg/test_mesh/test_column/test_Column.py` | `col_str.txt` |
| `test_dg/test_mesh/test_column/test_cell/test_Cell.py` | `cell_str.txt` |
| `test_dg/test_projection/test_Projection.py` | `mesh.png`, `mesh.json`, `proj.npy`, `proj_th.png`, `proj_xth.png`, `proj_yth.png`, `proj_xy.png`, `proj_xyth.png` |
| `test_utils/test_run_or_abort.py` | `driver.py` (the MPI driver it launches) |

`test_experiments/test_experiment_scripts.py` copies each `run.sh` into a `tempfile.TemporaryDirectory()`, which is deleted when the test ends. `test_dg/test_mesh/test_Mesh.py` writes a plot for every column and every cell, and takes much longer than the rest of the suite.

Regression Tests by Issue
--------------------------------------------------------------------------------

| Issue | Pull request | Test file(s) |
| --- | --- | --- |
| #6 (package `Error_Indicator` vs. imports of `error_indicator`) | #22 | `test_amr/test_imports.py` |
| #7, #8, #9 (*hp*-steering formulas, Eqs. 16, 18b, SI-6b) | #23 | `test_amr/test_hp_steering.py` |
| #10 (angular inflow term on non-conforming angular faces) | #24 | `test_rt/test_get_Eth.py` |
| #11 (error paths call `sys.exit`/`quit`; MPI ranks hang) | #25 | `test_dg/test_error_paths.py`, `test_utils/test_run_or_abort.py` |
| #12 (unsupported meshes accepted by `Problem.solve`) | #26 | `test_rt/test_solve.py` |
| #13 (Dirac boundary data written to the wrong elements) | #27 | `test_rt/test_boundary_conditions_vector.py` |
| #14 (fixed Krylov settings; unconverged solves used silently) | #28 | `test_rt/test_solve_krylov.py` |
| #15 (even-degree criterion not enforced by *p*-refinement) | #29 | `test_dg/test_mesh/test_even_degree.py` |
| #16 (uneven degree offset in the high-resolution reference mesh) | #30 | `test_amr/test_error_high_resolution.py` |
| #17 (under-resolved quadrature and stale normalization in `error_analytic`) | #31 | `test_amr/test_error_analytic.py` |
| #18 (`max_ntrial` off by one; `run.sh` error handling) | #32 | `test_experiments/test_experiment_scripts.py` |
| #19 (exported helpers that raised when called) | #33 | `test_dg/test_helpers.py` |

Source Files
--------------------------------------------------------------------------------

The source files contained in the `tests` subdirectory are:

- `conftest.py` : Puts `src` on `sys.path` and loads `pytest-mpi` (see above).

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `tests` subdirectory is

```
tests
|
|--- test_amr : Tests for src/amr (error indicators, hp-steering, package import).
|--- test_dg : Tests for src/dg (mesh, projection, masks, quadrature, error paths).
|    |--- test_mesh : Tests for the Mesh class and the even-degree criterion.
|    |    |--- test_column : Tests for the Column class.
|    |    |    |--- test_cell : Tests for the Cell class.
|    |--- test_projection : Tests for the Projection class.
|--- test_experiments : Tests for the experiment drivers and run.sh scripts.
|--- test_rt : Tests for src/rt (boundary data, angular inflow, solve checks, Krylov settings).
|--- test_utils : Tests for src/utils (run_or_abort under MPI).
```
