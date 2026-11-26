"""
Regression test for issue #11: an exception raised on the root rank while the
other ranks wait in a broadcast (as in the experiment drivers, which refine the
mesh on the root rank only) must abort the whole MPI job with a non-zero exit
status instead of hanging.
"""

# Standard Library Imports
import os
import shutil
import subprocess
import sys

# Third-Party Library Imports
import pytest

# Local Library Imports

SRC_DIR: str = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                             os.pardir, os.pardir, "src"))

DRIVER: str = """
import sys
sys.path.insert(0, {src_dir!r})

from mpi4py import MPI

from dg.mesh import Mesh
from utils import run_or_abort

def main():
    comm: MPI.Intracomm = MPI.COMM_WORLD
    mesh: Mesh = None
    if comm.rank == 0:
        mesh = Mesh(Ls = [1., 1.], pbcs = [False, False], ndofs = [2, 2, 2],
                    has_th = True)
        mesh.ref_col(0, kind = "spt", form = "q") # Unsupported form
    mesh = comm.bcast(mesh, root = 0)
    print("rank {{}} finished normally".format(comm.rank), flush = True)

if __name__ == "__main__":
    run_or_abort(main)
"""

@pytest.mark.mpi_skip # Launches its own mpirun
def test_run_or_abort_aborts_all_ranks(tmp_path):
    mpirun: str = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not found")

    driver_path: str = os.path.join(tmp_path, "driver.py")
    with open(driver_path, "w") as driver_file:
        driver_file.write(DRIVER.format(src_dir = SRC_DIR))

    # Do not let the child job inherit this process's MPI/PMIx environment
    env: dict = {key: val for key, val in os.environ.items()
                 if not key.startswith(("OMPI_COMM_WORLD", "OMPI_MCA_ess",
                                        "PMIX_", "PMI_"))}
    env["OMP_NUM_THREADS"] = "1"

    try:
        result = subprocess.run([mpirun, "--oversubscribe", "-n", "2",
                                 sys.executable, driver_path],
                                capture_output = True, text = True,
                                timeout = 120, env = env)
    except subprocess.TimeoutExpired:
        pytest.fail("MPI job hung after an exception on the root rank")

    output: str = result.stdout + result.stderr
    assert "UNSUPPORTED REFINEMENT FORM" in output, output
    assert "finished normally" not in output, output
    assert result.returncode != 0, output
