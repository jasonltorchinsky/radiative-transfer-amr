`consts`
================================================================================

This subdirectory contains definitions for constants used throughout the code, such as data types (e.g., `INT`, `REAL`) and numerical values (e.g., `PI`, `EPS`, `INF`, `COMM_ROOT`).

How It Fits Into the Whole
--------------------------------------------------------------------------------

`consts` is the bottom layer of the package: it imports only `numpy` and is imported (as `import consts`) by `utils`, `dg`, `rt`, `amr`, `tools`, the experiment drivers and the tests. Everything is defined in `__init__.py`; there are no other source files. The values are module-level names, not enforced constants: reassigning them (e.g., `consts.EPS = ...`) changes them for every module that reads them afterwards.

Constants
--------------------------------------------------------------------------------

The constants defined here are:

- `INT` (`type`, `numpy.int64`) : Integer data type. Used to cast index arrays, e.g., the column keys assigned to each MPI rank and the nonzero indices of sparse matrices (`np.array_split(...).astype(consts.INT)`).
- `REAL` (`type`, `numpy.float64`) : Floating-point data type. Used to define `EPS` and `INF`; other modules use the `numpy` default (`float64`) directly.
- `PI` (`float`, `numpy.pi`) : The numerical constant $\pi$. The angular domain is $[0, 2\,\mathrm{PI})$; for example, the root angular cell of a `Mesh` with `has_th = True` spans `[0, 2 * consts.PI]`.
- `EPS` (`float`, `numpy.finfo(REAL).resolution` $= 10^{-15}$) : The approximate decimal resolution of `REAL`, $10^{-\mathrm{precision}}$. It is neither the machine epsilon of `float64` ($\approx 2.2 \times 10^{-16}$) nor the smallest positive `float64`. It is the default relative and absolute tolerance of the Krylov solver in `rt.Problem.solve` (`rtol`, `atol`), and `rt.scattering_matrix` and `rt.preconditioner_matrix` skip a scattering contribution when the weighted coefficient $w_i w_j \sigma$ or the matrix entry has magnitude not exceeding the absolute threshold `10 * EPS`.
- `INF` (`float`, `numpy.finfo(REAL).max` $\approx 1.80 \times 10^{308}$) : The largest finite `REAL` (not `numpy.inf`). Used as an initial value for running minima and maxima (e.g., `-consts.INF` for maximum errors in `amr`), as the default refinement tolerances of `Error_Indicator` (`ref_tol = [INF, INF]`), for the condition number when the smallest singular value is zero, and, as `np.sqrt(INF)`, for the default divergence tolerance of the Krylov solver.
- `COMM_ROOT` (`int`, `0`) : The rank of the root MPI process. The root rank builds and refines meshes, computes error indicators, writes files and makes plots, and is the `root` of `mpi4py` broadcasts (`mpi_comm.bcast(obj, root = consts.COMM_ROOT)`).

Usage
--------------------------------------------------------------------------------

```python
import numpy as np
import consts

idxs = np.arange(10).astype(consts.INT)
th = np.linspace(0., 2. * consts.PI, 8, endpoint = False)

from mpi4py import MPI
if MPI.COMM_WORLD.Get_rank() == consts.COMM_ROOT:
    print("root rank")
```
