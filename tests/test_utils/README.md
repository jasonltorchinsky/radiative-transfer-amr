`test_utils`
================================================================================

This subdirectory contains unit tests for code in the `src/utils` subdirectory. `utils.run_or_abort(main)` runs `main()` and, if it raises, prints the traceback and calls `MPI.COMM_WORLD.Abort(1)`. Every experiment driver wraps its `main()` in it, because the drivers refine the mesh on the root rank only while the other ranks wait in a broadcast; without the abort, an exception on the root rank would leave the job hanging.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `test_utils` subdirectory are:

- `test_run_or_abort.py` : Issue #11 (PR #25), an exception on the root rank aborts the whole MPI job.

`test_run_or_abort.py`
--------------------------------------------------------------------------------

Regression test for issue #11 (the other half of the fix, `ValueError` instead of `sys.exit`/`quit`, is tested in `test_dg/test_error_paths.py`).

Module-level constants:

- `SRC_DIR` : absolute path of `<repository>/src`.
- `DRIVER` : the source of a small driver that mimics the experiment drivers. On rank 0 it builds a one-column `Mesh` and calls `mesh.ref_col(0, kind = "spt", form = "q")`, which raises `ValueError("... UNSUPPORTED REFINEMENT FORM ...")`; all ranks then call `comm.bcast(mesh, root = 0)` and would print `"rank <r> finished normally"`. `main()` is run through `run_or_abort`.

Test:

- `test_run_or_abort_aborts_all_ranks(tmp_path)` : writes the driver to `tmp_path/driver.py` and runs `mpirun --oversubscribe -n 2 <python> driver.py` with a 120 s timeout. It asserts that the job does not hang, that the output contains `"UNSUPPORTED REFINEMENT FORM"`, that no rank prints `"finished normally"`, and that the exit status is non-zero. The child job gets the current environment without the `OMPI_COMM_WORLD*`, `OMPI_MCA_ess*`, `PMIX_*` and `PMI_*` variables (so it does not attach to an enclosing MPI job) and with `OMP_NUM_THREADS=1`. The test is skipped if `mpirun` is not on `PATH`.

  It is marked `@pytest.mark.mpi_skip`: it starts its own `mpirun`, so it runs in a serial `pytest` session and is skipped under `--with-mpi`. The `mpirun` options are Open MPI's (`--oversubscribe`); other MPI implementations may need a different command.
