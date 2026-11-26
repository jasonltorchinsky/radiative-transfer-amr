"""
Regression tests for issue #18: small bugs in the experiment scripts.

- The drivers' `max_ntrial` stopping condition must allow exactly `max_ntrial`
  trials (it used `<` before the loop and `<=` inside it, giving one extra).
- The `run.sh` scripts must stop at the first failed step, report the failure
  in their exit status, and not call `mpirun` with an empty process count.
"""

# Standard Library Imports
import ast
import os
import pathlib
import shutil
import subprocess
import tempfile

# Third-Party Library Imports
import pytest

# Local Library Imports

EXPERIMENTS_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parents[2] / "experiments"

DRIVER_EXPERIMENTS: list = ["manufactured-solution", "one-weak-scatterer",
                            "two-strong-scatterers"]
RUN_SH_EXPERIMENTS: list = DRIVER_EXPERIMENTS + ["test-rt"]

## Fake `mpirun` and `python`: record the call, then exit with $FAKE_STATUS
FAKE_EXE: str = """#!/bin/bash
echo "$(basename "$0") $*" >> "${FAKE_LOG}"
exit "${FAKE_STATUS}"
"""


def keep_going_exprs(experiment: str) -> list:
    """
    Returns the compiled `keep_going = ...` expressions of an experiment
    driver, in source order (before the loop, then inside the loop).
    """
    driver_path: pathlib.Path = EXPERIMENTS_DIR / experiment / "experiment.py"
    tree: ast.Module = ast.parse(driver_path.read_text())

    exprs: list = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
            and node.target.id == "keep_going"):
            exprs += [(node.lineno, node.value)]
    exprs.sort(key = lambda item: item[0])

    return [compile(ast.Expression(value), str(driver_path), "eval")
            for _, value in exprs]


class FakeMesh:
    def get_ndof(self) -> int:
        return 1


@pytest.mark.parametrize("experiment", DRIVER_EXPERIMENTS)
@pytest.mark.parametrize("max_ntrial", [1, 2, 5])
def test_max_ntrial_counts_trials(experiment: str, max_ntrial: int):
    [keep_going_init, keep_going_update] = keep_going_exprs(experiment)

    namespace: dict = {"l_mesh": FakeMesh(),
                       "min_err": float("inf"),
                       "stopping_conditions": {"max_ndof": float("inf"),
                                               "max_ntrial": max_ntrial,
                                               "min_err": 0.}}

    ## Same loop structure as the driver
    ntrial: int = 0
    namespace["trial"] = 0
    keep_going: bool = eval(keep_going_init, namespace)
    while keep_going and ntrial <= max_ntrial:
        ntrial += 1
        namespace["trial"] += 1
        keep_going: bool = eval(keep_going_update, namespace)

    assert ntrial == max_ntrial


def run_script(experiment: str, args: list, status: int) -> tuple:
    """
    Runs an experiment's run.sh in a scratch directory with fake `mpirun` and
    `python` executables. Returns the completed process and the logged calls.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path: pathlib.Path = pathlib.Path(tmp_dir)
        shutil.copy(EXPERIMENTS_DIR / experiment / "run.sh", tmp_path / "run.sh")

        bin_path: pathlib.Path = tmp_path / "bin"
        bin_path.mkdir()
        for exe in ["mpirun", "python"]:
            (bin_path / exe).write_text(FAKE_EXE)
            (bin_path / exe).chmod(0o755)

        log_path: pathlib.Path = tmp_path / "calls.log"
        log_path.touch()

        env: dict = {"PATH": str(bin_path) + os.pathsep + os.environ.get("PATH", ""),
                     "FAKE_LOG": str(log_path),
                     "FAKE_STATUS": str(status)}
        result = subprocess.run(["bash", "run.sh"] + args, cwd = tmp_path,
                                env = env, capture_output = True, text = True,
                                timeout = 60)

        calls: list = log_path.read_text().splitlines()

    return (result, calls)


@pytest.mark.parametrize("experiment", RUN_SH_EXPERIMENTS)
def test_run_sh_stops_on_failure(experiment: str):
    [result, calls] = run_script(experiment, ["-n", "2"], status = 1)

    assert result.returncode != 0
    assert "experiment complete" not in result.stdout
    assert len(calls) == 1


@pytest.mark.parametrize("experiment", RUN_SH_EXPERIMENTS)
def test_run_sh_default_nprocs(experiment: str):
    [result, calls] = run_script(experiment, [], status = 0)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "experiment complete" in result.stdout
    assert len(calls) > 0
    for call in calls:
        tokens: list = call.split()
        assert tokens[0] == "mpirun"
        n_procs: str = tokens[tokens.index("-n") + 1]
        assert n_procs.isdigit() and int(n_procs) >= 1, call
