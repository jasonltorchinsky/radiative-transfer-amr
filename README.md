Angular-Spatial *hp*-Adaptivity for Radiative Transfer with Discontinuous Galerkin Spectral Element Methods
================================================================================

This code was used to generate the results reported in

J. L. Torchinsky, S. Du, and S. N. Stechmann, Angular-spatial *hp*-adaptivity for radiative transfer with discontinuous Galerkin spectral element methods, Journal of Quantitative Spectroscopy and Radiative Transfer 348 (2026) 109687, https://doi.org/10.1016/j.jqsrt.2025.109687.

The dataset associated with the publication is available at https://zenodo.org/records/14885262.

Equation numbers below refer to the published article; numbers prefixed with 'SI-' refer to its Supplemental Information.

Method Overview
--------------------------------------------------------------------------------

The code solves the steady-state, monochromatic radiative transfer equation (RTE) in two spatial dimensions and one angular dimension (Eq. 3),

$\vec{s} \cdot \nabla_{\vec{x}} u(\vec{x}, \vec{s}) + \kappa(\vec{x}) u(\vec{x}, \vec{s}) - \sigma(\vec{x}) \int_{\mathbb{S}} \Phi(\vec{s}, \vec{s}') u(\vec{x}, \vec{s}') \, d\vec{s}' = f(\vec{x}, \vec{s})$,

on $\Omega = [0, L_x] \times [0, L_y]$ with $\vec{s} = (\cos\theta, \sin\theta)$, $\theta \in [0, 2\pi)$, and inflow Dirichlet boundary conditions (Eq. 4).

- **Mesh.** The spatial domain is divided into rectangles ('columns', spatial *h*-elements $C$). Each column carries its own one-dimensional angular mesh of arcs ('cells', angular *h*-elements $K$). A spatial-angular element is a pair $(C, K)$ (Section 2.1). Both the spatial and angular meshes are kept 1-irregular, and neighboring elements obey the even-degree criterion (Section 3.2).
- **Basis.** On each $(C, K)$ the solution is a tensor product of Lagrange polynomials interpolating Legendre-Gauss-Lobatto (LGL) nodes in $x$ and $y$ and Legendre-Gauss (LG) nodes in $\theta$ (Eqs. 5-6). The unknowns are the nodal values $u^{C,K}_{ija}$.
- **Discretization.** An upwind discontinuous Galerkin (DG) spectral element method (Eqs. 10-12) yields a sparse linear system: mass (extinction) minus scattering plus boundary convection minus interior convection. The degrees of freedom on the inflow boundary are fixed by the boundary data, and the remaining (interior) system is solved with a PETSc Krylov solver through `petsc4py`, distributed over MPI ranks.
- **Adaptivity.** Spatial- and angular-jump error indicators (Eqs. 13-14) mark elements whose indicator exceeds a fraction of the maximum (Eq. 15; the code marks an element when its indicator is greater than or equal to the threshold). When both kinds of refinement are enabled (the Two Strong Scatterers driver), the driver chooses randomly between spatial and angular refinement with the probability of Eq. 16. An *hp*-steering criterion based on Legendre-coefficient decay (Eqs. 18a-18b, SI-3) chooses between *h*- and *p*-refinement for each marked element.

Code Layout
--------------------------------------------------------------------------------

All source code is in `src`. The packages import one another as top-level packages (e.g., `import consts`, `from dg.mesh import Mesh`), so `src` must be on `sys.path`; every package's `__init__.py` appends the appropriate directory when imported, and the experiment drivers and `tests/conftest.py` add `src` themselves.

| Package | Role |
| --- | --- |
| `src/consts` | Data types (`INT`, `REAL`) and numerical constants (`PI`, `EPS`, `INF`, `COMM_ROOT`). |
| `src/utils` | MPI-aware helpers: `print_msg` (timestamped output from the root rank) and `run_or_abort` (abort all ranks when a driver raises). |
| `src/dg` | The DG spectral element machinery: the `Mesh`/`Column`/`Cell` hierarchy and its refinement (`dg.mesh`), the `Projection` of a function onto the mesh (`dg.projection`), splitting and merging distributed PETSc matrices and vectors (`dg.matrix`), and 1-D quadrature and Lagrange polynomials (`dg.quadrature`). |
| `src/rt` | The `Problem` class, which holds $\kappa$, $\sigma$, $\Phi$, $f$ and the boundary data, assembles the DG matrices and vectors, and solves the linear system (`Problem.solve`). |
| `src/amr` | The `Error_Indicator` class: error indicators (spatial jump, angular jump, analytic, high-resolution, random), refinement marking, *hp*-steering and `ref_by_ind`, which returns the refined mesh. |
| `src/tools` | Plotting of meshes, projections, error indicators and matrices with `matplotlib`. |

One step of the solve and refinement loop, as written in `experiments/*/experiment.py`, runs as follows (all ranks execute the driver; "root" is rank `consts.COMM_ROOT` = 0):

1. Every rank holds the same `dg.mesh.Mesh` (built by the experiment's `mesh.py`, or broadcast from root after the previous refinement).
2. `rt.Problem.solve(mesh, **solver_params)` (collective) checks the mesh, assembles the mass, scattering, interior-convection and boundary-convection matrices and the forcing and boundary-condition vectors as PETSc objects distributed by columns, splits off the inflow-boundary DOFs (`dg.matrix`), and solves the interior system with a PETSc KSP. It returns `[uh, convergence_info, matrix_info]`, where `uh` is a `dg.projection.Projection` on root and `None` on the other ranks.
3. On root, `amr.error_indicator.Error_Indicator(uh, ...)` computes the angular-jump and/or spatial-jump indicators, marks elements and applies the *hp*-steering criterion. Analytic or high-resolution reference errors may also be computed (`error_high_resolution` solves a second problem and is collective).
4. On root, the driver (if it refines in both space and angle) picks spatial or angular refinement (Eq. 16) and calls `Error_Indicator.ref_by_ind()`, which returns a refined deep copy of the mesh (`Mesh.ref_col`, `Mesh.ref_cell`, which enforce 1-irregularity and the even-degree criterion).
5. The new mesh is broadcast to all ranks with `mpi4py`, the stopping criteria (maximum DOFs, maximum trials, minimum error) are checked, and the loop repeats.

Each driver wraps its `main()` in `utils.run_or_abort` so that an exception on root (e.g., during refinement) aborts the whole MPI job instead of leaving the other ranks waiting in the broadcast.

A minimal solve, run with `python` or `mpirun -n N python` from a directory where `src` is importable:

```python
import numpy as np
from dg.mesh import Mesh
from rt import Problem

mesh = Mesh(Ls = [1., 1.], pbcs = [False, False], ndofs = [3, 3, 3], has_th = True)
for _ in range(2):
    mesh.ref_mesh(kind = "ang", form = "h")

kappa = lambda x, y: 1.1 + 0. * x * y
sigma = lambda x, y: 0.1 + 0. * x * y
Phi   = lambda th, phi: (1. / (2. * np.pi)) + 0. * th * phi # Isotropic
f     = lambda x, y, th: 1. + 0. * x * y * th
bcs   = lambda x, y, th: 0. * x * y * th                    # No inflow
problem = Problem(kappa, sigma, Phi, f, [bcs, [None, None, None]])

[uh, convergence_info, matrix_info] = problem.solve(mesh) # uh is None off root
```

See `experiments/*/problem.py` for complete problem definitions and `src/rt/README.md` for the expected call signatures.

Installation
--------------------------------------------------------------------------------

Upon cloning this repository, you may obtain the `conda` environment from the provided `environment.yml` file via executing the following command in the root directory:

```conda env create -f environment.yml```

and activate it with

```conda activate radiative-transfer-amr```

The environment provides Python 3.12 with `numpy`, `scipy`, `matplotlib`, `mpi4py`, `petsc4py`, `pytest` and `pytest-mpi`.

There are several Bash scripts (files with extension `.sh`) included in this repository - especially in the `experiments` subdirectory - that are used to execute pre-written piece of code. To make such a script executable on your local machine, you use the `chmod` command, e.g.:

```chmod +x script_name.sh```

Running the Tests
--------------------------------------------------------------------------------

The unit tests in `tests` use `pytest`; `tests/conftest.py` puts `src` on `sys.path`. From the `tests` directory, run the serial tests with

```python -m pytest```

and the tests under MPI (with the `pytest-mpi` plugin) with

```mpirun -n 2 python -m pytest --with-mpi```

Tests marked `mpi_skip` (e.g., `test_utils/test_run_or_abort.py`, which launches its own `mpirun`) are skipped under `--with-mpi`.

Running the Experiments
--------------------------------------------------------------------------------

The numerical experiments of the publication (Manufactured Solution, Two Strong Scatterers, and One Weak Scatterer from Section SI-7) and a basic solver test (`test-rt`) are in `experiments`. Each experiment directory holds an `input.json` with its parameters, `mesh.py` and `problem.py` defining the initial mesh and the problem, `experiment.py` with the solve and refinement loop, plotting scripts, and a `run.sh` that runs everything with `N` MPI ranks:

```
cd experiments/manufactured-solution
./run.sh -n N
```

See `experiments/README.md` and the README file in each experiment directory for details.

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the root directory is

```
root
|
|--- experiments : Numerical experiments (as reported in the associated publication) and tests that require MPI.
|--- notes : Git submodule pointing to https://github.com/jasonltorchinsky/radiative-transfer-amr-notes (empty unless initialized with `git submodule update --init`); not needed to run the code.
|--- src : Source code for the implementation of the angular-spatial hp-adaptive algorithm for (steady-state, monochromatic) radiative transfer in two spatial dimensions and one angular dimension.
|--- tests : Unit tests that utilize the pytest testing framework.
```

The root directory also contains `environment.yml` (the `conda` environment), `LICENSE`, and `.gitmodules` (the `notes` submodule).
