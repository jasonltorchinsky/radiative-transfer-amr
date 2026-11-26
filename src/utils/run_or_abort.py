# Standard Library Imports
import sys
import traceback
from typing import Callable

# Third-Party Library Imports
from mpi4py import MPI

# Local Library Imports

# Relative Imports


def run_or_abort(main: Callable[[], None]) -> None:
    """
    Runs main() and aborts every MPI rank if it raises an exception.

    Without this, an exception on one rank (e.g., during refinement on the root
    rank) leaves the other ranks waiting in a collective operation forever.
    MPI_Abort also gives the job a non-zero exit status.
    """

    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)
