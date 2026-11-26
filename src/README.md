`src`
================================================================================

This subdirectory contains the source code implementing the angular-spatial *hp*-adaptive discontinuous Galerkin spectral element method (DGSEM) for the steady-state, monochromatic radiative transfer equation in two spatial dimensions and one angular dimension.

How the Packages Fit Together
--------------------------------------------------------------------------------

The packages form layers; each layer imports only from the layers above it in this list:

1. `consts` : constants and data types. Imported by almost every module.
2. `utils` : MPI-aware helpers (`print_msg`, `run_or_abort`). Imports `consts`.
3. `dg` : the spatial-angular *hp*-mesh (`dg.mesh`), functions represented on it (`dg.projection`), quadrature and Lagrange polynomials (`dg.quadrature`), and the distribution of global matrices and vectors over MPI ranks (`dg.matrix`). Imports `consts` and `utils`.
4. `rt` : the `Problem` class. It assembles the DG system for the radiative transfer equation (RTE; Eqs. 10-11 of the publication) from the `dg` building blocks and solves it with `petsc4py` (`Problem.solve`), returning the numerical solution as a `dg.projection.Projection`. Imports `consts`, `utils`, `dg`.
5. `amr` : the `Error_Indicator` class (in `amr.error_indicator`). It computes error indicators from a `Projection` (Eqs. 13-14), marks elements for refinement (Eq. 15), applies the *hp*-steering criteria (Eqs. 18a-18b, SI-3), and returns a refined `dg.mesh.Mesh`. `error_high_resolution` also uses an `rt.Problem` to solve on a finer mesh. Imports `consts`, `dg`, `rt`.
6. `tools` : plotting with `matplotlib` for meshes, projections, error indicators and matrices. Imports `consts`, `dg`, `amr`.

The experiment drivers in `experiments` combine these into the solve and refinement loop described in the root `README.md`: build a `Mesh`, call `Problem.solve`, build an `Error_Indicator` on the root rank, call `ref_by_ind`, broadcast the refined mesh, and repeat.

Package Initialization
--------------------------------------------------------------------------------

The modules import each other as top-level packages (`import consts`, `from dg.mesh import Mesh`, `from rt import Problem`), so the `src` directory itself must be on `sys.path`. `src/__init__.py`:

- defines `src_dir` (`str`), the absolute, normalized path of the `src` directory, and appends it to `sys.path` if it is not already there (side effect on import);
- imports the subpackages `consts`, `dg` and `utils`, so `import src` makes `src.consts`, `src.dg` and `src.utils` available.

`amr`, `rt` and `tools` are not imported by `src/__init__.py`; import them directly once `src` is on `sys.path` (e.g., `from rt import Problem`, `from amr.error_indicator import Error_Indicator`). `amr` has no `__init__.py` and is used as a namespace package. Each subpackage's `__init__.py` also appends a directory to `sys.path` when imported.

Typical use, from a script outside `src`:

```python
import os, sys
sys.path.append(os.path.normpath("path/to/radiative-transfer-amr/src"))

import consts
from dg.mesh import Mesh
from rt import Problem
from amr.error_indicator import Error_Indicator
```

Conventions Used Throughout
--------------------------------------------------------------------------------

- The spatial domain is $[0, L_x] \times [0, L_y]$ and the angular domain is $\theta \in [0, 2\pi)$, with direction $\vec{s} = (\cos\theta, \sin\theta)$.
- A spatial element is a `Column` and an angular element within a column is a `Cell`; a spatial-angular element $(C, K)$ is a (column, cell) pair. Columns are stored in `Mesh.cols` and cells in `Column.cells`, both as dictionaries keyed by an integer key computed from the element's index and refinement level.
- Quadrature nodes and basis functions are defined on the reference interval $[-1, 1]$ (`dg.quadrature`) and mapped to physical elements with `dg.projection.push_forward` and `dg.projection.pull_back`.
- Within a spatial-angular element with $n_x \times n_y \times n_\theta$ degrees of freedom, values are stored as arrays of shape `[nx, ny, nth]` indexed `[i, j, a]`, and flattened with $\theta$ fastest: $\alpha = n_\theta n_y i + n_\theta j + a$ (`dg.matrix.get_idx_map`). Global vectors list leaf columns in increasing key order, and within each column the leaf cells in increasing key order.
- MPI: all ranks call the assembly and solve routines (they are collective); mesh refinement, error indicators, file output and plotting are done on rank `consts.COMM_ROOT` (0), and the refined mesh is then broadcast.

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `src` subdirectory is

```
src
|
|--- amr : Contains source code regarding adaptive mesh refinement, such as the Error_Indicator class, different ways to calculate error (e.g., spatial jump, angular jump, analytic, high-resolution), hp-steering, and refining a mesh based on an error indicator.
|--- consts : Contains definitions for constants used throughout the code, such as data types (e.g., INT, REAL) and numerical values (e.g., PI, EPS, INF, COMM_ROOT).
|--- dg : Contains source code regarding discontinuous Galerkin methods, such as the Mesh and Projection classes, methods for numerical quadrature, and methods for handling matrices across multiple MPI processes.
|--- rt : Contains source code regarding the numerical algorithm for solving the steady-state, monochromatic radiative transfer equation, including constructing the extinction (mass), scattering, and convection matrices, the Problem class, and solving the matrix system that arises.
|--- tools : Contains source code for generating relevant plots, e.g., of the *hp*-mesh, numerical solutions on the *hp*-mesh, error indicators, and matrices.
|--- utils : Contains source code for utility functions, such as printing messages to the command terminal from the root MPI process and aborting all MPI processes when one raises an exception.
```
