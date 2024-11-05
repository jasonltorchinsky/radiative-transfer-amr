import os, sys
src_dir: str = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, "src"))

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Standard Library Imports
import pathlib
import shutil
import tempfile

# Third-Party Library Imports
import pytest, pytest_mpi
from mpi4py import MPI

# Local Library Imports

#@pytest.fixture(name="mpi_tmp_path_fixed")
#def mpi_tmp_path_fixed_fixture():
#    """
#    Exposes pytest-mpi logic but overrides pytest's temporary file management.
#    """
#
#    ## Initialize MPI
#    if not MPI.Is_initialized():
#        MPI.init()
#    mpi_comm: MPI.Intracomm = MPI.COMM_WORLD
#
#    temp = tempfile.mkdtemp()
#    yield pytest_mpi.mpi_tmp_path.__wrapped__(pathlib.Path(temp))
#    if mpi_comm.Get_rank() == 0:
#        shutil.rmtree(temp)