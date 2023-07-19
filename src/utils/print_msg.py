import petsc4py
from   datetime import datetime
from   mpi4py   import MPI
from   petsc4py import PETSc

def print_msg(msg, **kwargs):
    """
    Prints the given message with the current time.
    """
    
    default_kwargs = {'blocking' : True # Synchronize ranks before exiting
                      } 
    kwargs = {**default_kwargs, **kwargs}
    
    # Initialize parallel communicators
    MPI_comm = MPI.COMM_WORLD
    
    petsc4py.init()
    comm      = PETSc.COMM_WORLD
    comm_rank = comm.getRank()
    comm_size = comm.getSize()

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    
    out_msg = ('[{}]: {}').format(current_time, msg)

    if comm_rank == 0:
        PETSc.Sys.Print(out_msg)

    if kwargs['blocking']:
        MPI_comm.Barrier()
    
    return None
