`amr`
================================================================================

This subdirectory contains the source code regarding adaptive mesh refinement (AMR). It turns a numerical solution $u_{hp}$ on a spatial-angular *hp*-mesh into per-element error indicators, decides which elements to refine and whether to refine them in *h* or in *p*, and produces the refined mesh for the next solve.

How It Fits Into the Package
--------------------------------------------------------------------------------

The code uses the mesh terminology of the `dg` package: a *column* (`col`) is a spatial *hp*-element $C$ (a rectangle $[x_0, x_1] \times [y_0, y_1]$), and a *cell* is an angular *hp*-element $K$ (an interval $[\theta_0, \theta_1] \subset [0, 2\pi]$) inside a column. A spatial-angular *hp*-element is a pair $(C, K)$. Columns and cells are identified by the integer keys used in `dg.mesh.Mesh.cols` and `Column.cells`.

One step of the adaptive loop (Section 3.2 of the paper, steps 1-7) is split between the packages as follows:

1. `rt.Problem.solve(mesh)` returns the numerical solution as a `dg.projection.Projection` (on the MPI root rank only).
2. `amr.error_indicator.Error_Indicator(uh, ref_kind, ref_form, ref_tol)` wraps that solution. One of its `error_*` methods (in `error_indicator/error`) computes the column and cell error indicators, applies the refinement thresholds (Eq. 15), marks elements, and, for `ref_form = "hp"`, calls the *hp*-steering criteria `col_hp_steer` (SI Eq. 6b) and `cell_hp_steer` (Eq. 18b).
3. When both spatial and angular refinement are considered, the experiment driver compares the averages returned by `Error_Indicator.mean_error_to_resolve()` of a spatial and an angular indicator to apply the spatial-angular steering criterion (Eq. 16). The random draw itself is made in the driver (`experiments/two-strong-scatterers/experiment.py`), not in this package.
4. `Error_Indicator.ref_by_ind()` returns a refined copy of the mesh by calling `dg.mesh.Mesh.ref_cell` and `dg.mesh.Mesh.ref_col`, which also enforce 1-irregularity and the even-degree criterion.
5. The new mesh is broadcast to all ranks and solved again.

`Error_Indicator.to_file` and `amr.error_indicator.from_file` save and reload an indicator so that plots can be made after a run (`experiments/*/generate_experiment_visualizations.py`, `src/tools/amr`).

This package calls `consts`, `dg.mesh`, `dg.projection`, `dg.quadrature`, and (for the high-resolution error only) `rt.Problem`, `mpi4py` and `petsc4py`. It is used by `experiments/*/experiment.py`, `experiments/*/generate_experiment_visualizations.py`, `src/tools/amr/plot_error_indicator.py`, `src/tools/amr/plot_refinement_thresholds.py`, and `tests/test_amr`.

Importing
--------------------------------------------------------------------------------

`amr` has no `__init__.py`; it is used as a namespace package. Put `src` on `sys.path` and import from the subpackage:

```python
import sys
sys.path.append("path/to/src")

from amr.error_indicator import Error_Indicator, from_file
```

Paper-to-Code Map
--------------------------------------------------------------------------------

| Paper | Code |
| --- | --- |
| Spatial-jump error indicator $\eta_{\vec{x}}^{C}$, Eq. (13) | `Error_Indicator.error_spatial_jump` |
| Angular-jump error indicator $\eta_{\vec{s}}^{(C,K)}$, Eq. (14) | `Error_Indicator.error_angular_jump` |
| Refinement thresholds $T_{\vec{x}}$, $T_{\vec{s}}$, Eq. (15) | marking block at the end of every `error_*` method |
| Spatial and angular EA2R and probability $p_{\vec{x}}$, Eq. (16) | `Error_Indicator.mean_error_to_resolve` (averages); draw in the experiment driver |
| Angular *hp*-steering, Eqs. (17)-(19) | `Error_Indicator.cell_hp_steer` |
| Spatial *hp*-steering, SI Eqs. (1)-(7) | `Error_Indicator.col_hp_steer` |
| Spatial and angular *h*- and *p*-refinement, 1-irregularity, even-degree criterion | `Error_Indicator.ref_by_ind`, which calls `dg.mesh.Mesh.ref_cell` and `dg.mesh.Mesh.ref_col` |

The analytic, high-resolution and random errors are not error indicators of the paper's algorithm. The analytic and high-resolution errors measure the accuracy of $u_{hp}$ for the reported error histories, and the random error is a test device.

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `amr` subdirectory is

```
amr
|
|--- error_indicator : Contains source code implementing the error indicator, of especial note is the Error_Indicator class, its hp-steering and refinement methods, and its file input/output.
     |
     |--- error : The methods that compute the error indicators (analytic, angular jump, spatial jump, high-resolution, random) and mark elements for refinement.
     |--- error_indicator_column : The Error_Indicator_Column class, which stores the indicator of one spatial element (column).
          |
          |--- error_indicator_cell : The Error_Indicator_Cell class, which stores the indicator of one angular element (cell).
```
