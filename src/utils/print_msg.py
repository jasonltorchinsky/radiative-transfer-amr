# Standard Library Imports
from datetime import datetime

# Third-Party Library Imports
import petsc4py

from mpi4py   import MPI
from petsc4py import PETSc

# Local Library Imports

# Relative Imports


def print_msg(msg: str, **kwargs) -> None:
    """
    Prints the given message with the current time.
    """
    
    default_kwargs: dict = {"blocking" : True # Synchronize ranks before exiting
                      } 
    kwargs: dict = {**default_kwargs, **kwargs}
    
    # Initialize parallel communicators
    MPI_comm: MPI.Intracomm = MPI.COMM_WORLD
    
    if not PETSc.Sys.isInitialized():
        petsc4py.init(comm = MPI_comm)
    PETSc_comm: PETSc.Comm = PETSc.COMM_WORLD
    comm_rank: int = PETSc_comm.getRank()
    comm_size: int = PETSc_comm.getSize()

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    
    out_msg: str = ("[{}]: {}").format(current_time, msg)

    if comm_rank == 0:
        PETSc.Sys.Print(out_msg)

    if kwargs["blocking"]:
        MPI_comm.Barrier()