`utils`
================================================================================

This subdirectory contains MPI-aware utility functions: printing a timestamped message from the root MPI process (`print_msg`) and running a driver's `main()` so that an exception on any rank aborts the whole MPI job (`run_or_abort`).

How It Fits Into the Whole
--------------------------------------------------------------------------------

`utils` depends only on `consts`, `mpi4py` and `petsc4py`. It is imported as `import utils` or `from utils import ...` (with `src` on `sys.path`).

- `print_msg` is called by the matrix and vector assembly routines in `rt` (`mass_matrix`, `scattering_matrix`, `preconditioner_matrix`, `interior_convection_matrix`, `boundary_convection_matrix`, `forcing_vector`, `boundary_conditions_vector`), by `rt.Problem.solve`, and by `dg.matrix.get_masks`, in each case only when the caller is passed `verbose = True`.
- `run_or_abort` wraps the `main()` function of every experiment driver (`experiments/*/experiment.py`). In those drivers the root rank refines the mesh while the other ranks wait in `mpi_comm.bcast`; if refinement raises on the root rank, `run_or_abort` aborts all ranks instead of leaving them waiting forever.

Package Initialization
--------------------------------------------------------------------------------

`__init__.py` re-exports `print_msg` and `run_or_abort`, so `from utils import print_msg, run_or_abort` works. It also defines `src_dir` (`str`) and appends it to `sys.path` if absent. `src_dir` is computed as `os.path.join(os.path.dirname(__file__), os.pardir, "src")`, which resolves to `src/src`, a directory that does not exist; the other packages' `__init__.py` files compute `src` itself. Because `import utils` as a top-level package already requires `src` to be on `sys.path`, this has no practical effect.

Functions
--------------------------------------------------------------------------------

### `print_msg(msg: str, **kwargs) -> None`

Defined in `print_msg.py`. Prints `msg` prefixed by the current wall-clock time, in the form `[HH:MM:SS]: msg`, from the root MPI process.

Arguments:
- `msg` (`str`) : The message to print. It is printed as given; include a trailing `"\n"` if needed (callers in `rt` do).
- `blocking` (`bool`, keyword, default `True`) : If `True`, call `MPI.COMM_WORLD.Barrier()` before returning. Other keyword arguments are accepted and ignored, so callers may pass their own `**kwargs` through.

Returns `None`.

Side effects:
- Initializes `petsc4py` on `MPI.COMM_WORLD` if PETSc has not yet been initialized.
- Prints with `PETSc.Sys.Print` only on the rank with `PETSc.COMM_WORLD.getRank() == consts.COMM_ROOT`.
- With `blocking = True` (the default) it is a collective operation: every rank of `MPI.COMM_WORLD` must call it, otherwise the job hangs at the barrier. Pass `blocking = False` to print from code that only some ranks execute.

Example:

```python
import utils
utils.print_msg("Assembling the mass matrix...\n")                    # all ranks call this
utils.print_msg("Refining on the root rank.\n", blocking = False)     # safe on one rank
```

### `run_or_abort(main: Callable[[], None]) -> None`

Defined in `run_or_abort.py`. Calls `main()`. If `main()` raises an `Exception`, prints the traceback to `stderr`, flushes `stdout` and `stderr`, and calls `MPI.COMM_WORLD.Abort(1)`, which terminates every rank of the job and gives it a non-zero exit status.

Arguments:
- `main` (`Callable[[], None]`) : The function to run; it takes no arguments and its return value is discarded.

Returns `None` when `main()` returns normally; does not return if `main()` raises an `Exception`.

Notes:
- Only subclasses of `Exception` are caught. `SystemExit` (e.g., from `sys.exit` or `argparse` errors) and `KeyboardInterrupt` propagate normally.
- Without this wrapper, an exception on one rank while the other ranks wait in a collective operation (such as the `bcast` of the refined mesh in the experiment drivers) leaves the job waiting indefinitely. The regression test is `tests/test_utils/test_run_or_abort.py`.

Example (the pattern used by every experiment driver):

```python
from utils import run_or_abort

def main():
    ... # solve, compute error indicators and refine on the root rank, broadcast the mesh

if __name__ == "__main__":
    run_or_abort(main)
```

Source Files
--------------------------------------------------------------------------------

- `__init__.py` : Adds a path to `sys.path` and re-exports `print_msg` and `run_or_abort`.
- `print_msg.py` : Defines `print_msg`.
- `run_or_abort.py` : Defines `run_or_abort`.
