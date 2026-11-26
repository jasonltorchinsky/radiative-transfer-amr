"""
Regression tests for issue #14: the Krylov solver settings used by Problem.solve
must be configurable (keyword arguments, e.g., solver_params in input.json, and
the PETSc options database), keep the previous values as defaults, and a solve
that does not converge must be reported instead of being used silently.
"""

# Standard Library Imports
import warnings

# Third-Party Library Imports
import numpy as np
from petsc4py import PETSc

# Local Library Imports
import consts
from amr.error_indicator import Error_Indicator
from dg.mesh import Mesh
from dg.projection import Projection
from rt import Problem

# Relative Imports

def get_problem_and_mesh() -> list:
    ## A small problem with extinction, scattering, and a source, for which the
    ## default solver settings converge in a few iterations on one or two ranks
    def kappa(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 10. + 0. * x * y
    def sigma(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 0.1 * kappa(x, y)
    def Phi(th: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return (1. / (3. * consts.PI)) * (1. + (np.cos(th - phi))**2)
    def f(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> np.ndarray:
        return (1. + x * y) * (1. + np.cos(th)**2)
    def bcs(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> np.ndarray:
        return 0. * x * y * th
    bcs_dirac: list = [bcs, [None, None, None]]
    problem: Problem = Problem(kappa, sigma, Phi, f, bcs_dirac)

    mesh: Mesh = Mesh(Ls = [1., 1.], pbcs = [False, False],
                      ndofs = [3, 3, 3], has_th = True)

    return [problem, mesh]

def not_converged_warnings(caught: list) -> list:
    return [w for w in caught if issubclass(w.category, RuntimeWarning)
            and "did not converge" in str(w.message)]

def test_solve_default_ksp_settings():
    [problem, mesh] = get_problem_and_mesh()

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        [_, info, _] = problem.solve(mesh)

    ## Defaults are the previously hard-coded values
    assert(info["rtol"] == consts.EPS)
    assert(info["atol"] == consts.EPS)
    assert(info["divtol"] == np.sqrt(consts.INF))
    assert(info["max_it"] == 5000)
    assert(info["gmres_restart"] == 50)

    assert(info["converged_reason"] > 0)
    assert(info["converged"])
    assert(len(not_converged_warnings(caught)) == 0)

def test_solve_ksp_settings_from_kwargs():
    [problem, mesh] = get_problem_and_mesh()
    comm_rank: int = PETSc.COMM_WORLD.getRank()

    ## Stop after two iterations, which cannot reach the tolerance
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        [_, info, _] = problem.solve(mesh, max_it = 2, gmres_restart = 30)

    assert(info["max_it"] == 2)
    assert(info["gmres_restart"] == 30)
    assert(info["iteration_number"] == 2)
    assert(info["converged_reason"] == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT)
    assert(not info["converged"])
    if comm_rank == consts.COMM_ROOT:
        assert(len(not_converged_warnings(caught)) == 1)

    ## A loose relative tolerance stops early with KSP_CONVERGED_RTOL
    [_, info, _] = problem.solve(mesh, rtol = 1.e-4, atol = 0.)
    assert(info["rtol"] == 1.e-4)
    assert(info["atol"] == 0.)
    assert(info["converged_reason"] == PETSc.KSP.ConvergedReason.CONVERGED_RTOL)
    assert(info["res_f"] <= 1.e-4 * info["convergence_history"][0])

def test_solve_ksp_settings_from_petsc_options():
    [problem, mesh] = get_problem_and_mesh()

    opts: PETSc.Options = PETSc.Options()
    opts["ksp_max_it"] = 3
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            [_, info, _] = problem.solve(mesh)
    finally:
        opts.delValue("ksp_max_it")

    assert(info["max_it"] == 3)
    assert(info["iteration_number"] == 3)
    assert(not info["converged"])

def test_error_high_resolution_forwards_ksp_settings():
    [problem, mesh] = get_problem_and_mesh()

    def u0(x: np.ndarray, y: np.ndarray, th: np.ndarray) -> np.ndarray:
        return 0. * x * y * th
    err_ind: Error_Indicator = Error_Indicator(Projection(mesh, u0))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        [_, info, _] = err_ind.error_high_resolution(problem,
                                                     ksp_type = "gmres",
                                                     pc_type = "bjacobi",
                                                     max_it = 2)

    assert(info["max_it"] == 2)
    assert(info["iteration_number"] == 2)
    assert(not info["converged"])
