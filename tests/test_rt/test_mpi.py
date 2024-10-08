# Standard Library Imports
import os

# Third-Party Library Imports
from mpi4py import MPI
import pytest, pytest_mpi

# Local Library Imports

# Relative Imports

#@pytest.fixture
#def mpi_tmp_path(tmp_path):
#    breakpoint()
#    return pytest_mpi.mpi_tmp_path(tmp_path)

@pytest.mark.mpi()
def test_mpi(mpi_tmp_path):
    ## Initialize MPI
    if not MPI.Is_initialized():
        MPI.init()
    MPI_comm: MPI.Intracomm = MPI.COMM_WORLD
    comm_rank: int = MPI_comm.Get_rank()

    ## Write comm info to file
    file_name: str = "rank_{}.txt".format(comm_rank)
    file_path: str = os.path.join(mpi_tmp_path, file_name)

    with open(file_path, "w") as file:
        file.write("{}".format(comm_rank))

    if MPI.Is_initialized():
        MPI.Finalize()