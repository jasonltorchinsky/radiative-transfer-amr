`error_indicator`
================================================================================

This subdirectory contains the source code regarding the error indicator. The `Error_Indicator` class holds a numerical solution, the refinement settings, and the per-column and per-cell error indicators; its methods compute the indicators, choose between *h*- and *p*-refinement, refine the mesh, and write the indicator to file.

How It Fits Into the Package
--------------------------------------------------------------------------------

An `Error_Indicator` is built from a `dg.projection.Projection` returned by `rt.Problem.solve`. The usual sequence on the MPI root rank is:

```python
from amr.error_indicator import Error_Indicator

err_ind = Error_Indicator(uh, ref_kind = "ang", ref_form = "hp", ref_tol = [0.8, 0.0])
err_ind.error_angular_jump()          # compute indicators, mark, hp-steer
err_ind.to_file("err_ind_ang_jmp.json", write_projection = False, write_mesh = False)
new_mesh = err_ind.ref_by_ind()       # refined copy of uh.mesh
```

The experiment drivers then broadcast `new_mesh` to all ranks and call `problem.solve(new_mesh)` again. The `error_*` methods are defined in the `error` subdirectory and attached to the class; the per-element containers are defined in `error_indicator_column` and `error_indicator_column/error_indicator_cell`.

Conventions
--------------------------------------------------------------------------------

- **Keys.** `self.cols` is a dict keyed by the integer column keys of `self.proj.mesh.cols`; `self.cols[col_key].cells` is keyed by the integer cell keys of that column. Only leaf columns and cells appear; the code asserts `is_lf` on every element it visits. Loops run in sorted key order.
- **Arrays.** Nodal values of a cell are `proj.cols[col_key].cells[cell_key].vals`, shape `[nx, ny, nth]`, index order $(x, y, \theta)$. Spatial nodes are Legendre-Gauss-Lobatto (LGL) and angular nodes are Legendre-Gauss (LG), both in ascending order on the reference interval $[-1, 1]$ (`dg.quadrature.quad_xyth`).
- **Refinement kind** `ref_kind`: `"spt"` marks only columns, `"ang"` marks only cells, `"all"` marks both. Any other value marks nothing (no error is raised).
- **Refinement form** `ref_form`: `"h"` or `"p"` is copied to every marked element; `"hp"` selects `"h"` or `"p"` per element with `col_hp_steer` or `cell_hp_steer`. Other strings are not checked here; `dg.mesh.Mesh.ref_cell` raises `ValueError` on them during `ref_by_ind`.
- **Refinement tolerances** `ref_tol = [ang_ref_tol, spt_ref_tol]`, angular first. The thresholds are $T_{\vec{s}} = \tau_{\vec{s}} \max_{(C,K)} \eta^{(C,K)}$ and $T_{\vec{x}} = \tau_{\vec{x}} \max_C \eta^{C}$ (Eq. 15). An element is marked when its indicator is `>=` the threshold. The paper says "exceeds"; with `>=`, a tolerance of 0 marks every element, including elements with zero error, and, for non-negative indicators, a tolerance of at most 1 always marks the element with the largest indicator. The default `[consts.INF, consts.INF]` (`consts.INF` is the largest finite `REAL`) gives a threshold above any positive error, so nothing is marked unless every indicator is 0.
- **MPI.** Except for `error_high_resolution`, nothing here communicates. The experiment drivers create and use `Error_Indicator` objects on `consts.COMM_ROOT` only, because `Problem.solve` returns the solution only there. `error_high_resolution` must be called on every rank.

Exported Names
--------------------------------------------------------------------------------

`__init__.py` exports:

- `Error_Indicator` : the class defined in `class_Error_Indicator.py`.
- `from_file` : the reader defined in `from_file.py`.
- `error` : the subpackage of error-computing methods.
- `error_indicator_column` : the subpackage containing `Error_Indicator_Column` (and, inside it, `error_indicator_cell` with `Error_Indicator_Cell`).

`Error_Indicator`
--------------------------------------------------------------------------------

```python
Error_Indicator(proj: Projection, ref_kind: str = "all", ref_form: str = "hp",
                ref_tol: list = [consts.INF, consts.INF])
```

Arguments:

- `proj` (`dg.projection.Projection`) : the numerical solution $u_{hp}$ and, through `proj.mesh`, its mesh. Stored by reference, not copied.
- `ref_kind` (`str`, default `"all"`) : `"spt"`, `"ang"` or `"all"`, see Conventions.
- `ref_form` (`str`, default `"hp"`) : `"h"`, `"p"` or `"hp"`, see Conventions.
- `ref_tol` (`list` of two floats, default `[consts.INF, consts.INF]`) : `[ang_ref_tol, spt_ref_tol]`, the $\tau_{\vec{s}}$ and $\tau_{\vec{x}}$ of Eq. (15). The paper uses values in $[0, 1]$.

The constructor only stores its arguments. The experiment drivers pass the refinement-strategy entries of `input.json` directly, e.g. `Error_Indicator(uh, **{"ref_kind": "ang", "ref_form": "hp", "ref_tol": [0.8, 0.0]})`.

Attributes set by the constructor: `proj`, `ref_kind`, `ref_form`, `ang_ref_tol`, `spt_ref_tol`.

Attributes set by any `error_*` method (and by `from_file`):

- `cols` (`dict[int, Error_Indicator_Column]`) : the indicator of every leaf column, each holding the indicators of its leaf cells.
- `error` (`float`) : the global error (its definition depends on the method; see `error/README.md`).
- `col_max_error`, `cell_max_error` (`float`) : the largest column and cell indicators, used for the thresholds of Eq. (15). They are `-consts.INF` when there is nothing to take the maximum of (e.g. `cell_max_error` after `error_spatial_jump`).
- `error_to_resolve` (`float`) : the sum of the indicators of all marked columns and cells. `mean_error_to_resolve()` divides it by the number of marked elements.

Methods (each documented below or in `error/README.md`):

| Method | Defined in | Purpose |
| --- | --- | --- |
| `error_analytic(anl_sol, quad_rtol = 1.e-2, max_quad_dbls = 4)` | `error/error_analytic.py` | Relative $L^2$ error against an analytic solution |
| `error_angular_jump()` | `error/error_angular_jump.py` | Angular-jump indicator, Eq. (14) |
| `error_spatial_jump()` | `error/error_spatial_jump.py` | Spatial-jump indicator, Eq. (13) |
| `error_high_resolution(problem, uh_hr = None, **kwargs)` | `error/error_high_resolution.py` | Relative $L^2$ difference from a *p*-enriched solution (MPI collective) |
| `error_random(rng = None)` | `error/error_random.py` | Random indicator, for testing |
| `col_hp_steer(col_key)` | `col_hp_steer.py` | Spatial *hp*-steering, SI Eq. (6b) |
| `cell_hp_steer(col_key, cell_key)` | `cell_hp_steer.py` | Angular *hp*-steering, Eq. (18b) |
| `mean_error_to_resolve()` | `mean_error_to_resolve.py` | Average indicator of the marked elements, Eq. (16) |
| `ref_by_ind()` | `ref_by_ind.py` | Refined copy of the mesh |
| `to_file(file_path = "err_ind.json", **kwargs)` | `to_file.py` | Write the indicator as JSON |

Marking and Steering in the `error_*` Methods
--------------------------------------------------------------------------------

Every `error_*` method ends with the same block. It sets `error_to_resolve = 0`, computes $T_{\vec{s}}$ = `ang_ref_tol * cell_max_error` and $T_{\vec{x}}$ = `spt_ref_tol * col_max_error`, and then, for each leaf column in sorted key order:

- if `ref_kind` is `"spt"` or `"all"`: if `cols[col_key].error >= ` $T_{\vec{x}}$, set `do_ref = True`, add the column indicator to `error_to_resolve`, and set `ref_form` to `col_hp_steer(col_key)` when `self.ref_form == "hp"` and to `self.ref_form` otherwise; else set `do_ref = False`;
- if `ref_kind` is `"ang"` or `"all"`: the same for each leaf cell of the column, with $T_{\vec{s}}$ and `cell_hp_steer(col_key, cell_key)`.

Elements that are not marked keep `ref_form = ""`. The *hp*-steering functions are therefore evaluated only for marked elements.

`col_hp_steer`
--------------------------------------------------------------------------------

```python
Error_Indicator.col_hp_steer(self, col_key: int) -> str
```

Spatial *hp*-steering criterion of SI Section SI-3. Returns `"p"` or `"h"` for the leaf column `col_key` of `self.proj`. It reads `self.proj` only and has no side effects.

Steps, with $n_x, n_y$ = `col.ndofs`, LGL nodes and weights $(x_i, w_i)$, $(y_j, w_j)$ on $[-1, 1]$:

1. $g_{ij} = g(x_i, y_j)$ with $g(\vec{x}) = \sum_{K \in C} \int_K u_{hp}(\vec{x}, \theta) \, d\theta$ (SI Eq. 7), from `Projection.col_intg_th(col_key)`, an array of shape `[nx, ny]`. The angular integral is in physical $\theta$ (Jacobian $\Delta\theta / 2$ included).
2. Legendre coefficients $a_{pq} = c^{x}_p c^{y}_q \sum_{i,j} w_i w_j g_{ij} L_p(x_i) L_q(y_j)$, computed only for the top rows $p = n_x - 1$ (all $q$) and $q = n_y - 1$ (all $p$). The normalization is $c_k = (2k+1)/2$ for $k < n - 1$ and $c_{n-1} = (n-1)/2$ for the top index.
3. $|\gamma^{C,x}_f|^2 = \sum_{q=0}^{n_y-1} \frac{2}{2q+1} a_{n_x-1, q}^2$ and $|\gamma^{C,y}_f|^2 = \sum_{p=0}^{n_x-1} \frac{2}{2p+1} a_{p, n_y-1}^2$ (SI Eq. 6a).
4. $lp_{\vec{x}} = \frac{1}{2} \left( \frac{\log\left( (2n_x-1) / (2|\gamma^{C,x}_f|^2) \right)}{2 \log(n_x - 1)} + \frac{\log\left( (2n_y-1) / (2|\gamma^{C,y}_f|^2) \right)}{2 \log(n_y - 1)} \right)$ (SI Eq. 6b).
5. Return `"p"` if $lp_{\vec{x}} - \frac{1}{2} \ge \frac{1}{2}(n_x + n_y)$, otherwise `"h"`.

Difference from the paper: SI Eq. (5) uses the factor $(2p+1)(2q+1)/4$ for every index. The code replaces $(2k+1)/2$ by $(n-1)/2$ at the top index $k = n - 1$, because the $n$-point LGL rule gives $\sum_i w_i L_{n-1}(x_i)^2 = 2/(n-1)$ rather than $2/(2n-1)$; the code comment states that this gives the exact Legendre coefficients of the nodal interpolant of $g$. For the top index the two normalizations differ by the factor $(2n-1)/(n-1)$.

Edge cases (the same holds for the paper's formula): for $n_x = 2$ or $n_y = 2$ the denominator $\log(n - 1)$ is 0, and for $n = 1$ it is $-\infty$; numpy then returns $\pm\infty$ or `nan` with a `RuntimeWarning`, and a `nan` comparison selects `"h"`. A zero top coefficient gives $lp = +\infty$ and selects `"p"`. Because $g$ is an integral in physical units, $|\gamma_f|^2$ scales with the magnitude of $u_{hp}$, so the decision is not invariant under rescaling the solution.

`cell_hp_steer`
--------------------------------------------------------------------------------

```python
Error_Indicator.cell_hp_steer(self, col_key: int, cell_key: int) -> str
```

Angular *hp*-steering criterion of Section 3.4 of the paper. Returns `"p"` or `"h"` for the leaf cell `cell_key` of the leaf column `col_key`. It reads `self.proj` only and has no side effects.

Steps, with $n_\theta$ = `cell.ndofs[0]`, LG nodes and weights $(\theta_a, w_a)$ on $[-1, 1]$:

1. $g_a = \int_{\Omega_C} u_{hp}(\vec{x}, \theta_a) \, d\vec{x}$ (Eq. 19), from `Projection.cell_intg_xy(col_key, cell_key)`, an array of length `nth`. The spatial integral is in physical coordinates (Jacobian $\Delta x \Delta y / 4$ included).
2. $\gamma_f = \frac{2n_\theta - 1}{2} \sum_a w_a g_a L_{n_\theta-1}(\theta_a)$ (Eq. 18a with $r = n_\theta - 1$; the LG rule integrates $L_{n_\theta-1}^2$ exactly).
3. $lp_\theta = \log\left( (2n_\theta-1) / (2\gamma_f^2) \right) / \left( 2 \log(n_\theta - 1) \right)$ (Eq. 18b, with the squared top coefficient $|\gamma_f|^2$).
4. Return `"p"` if $lp_\theta - \frac{1}{2} \ge n_\theta$, otherwise `"h"`.

The edge cases listed for `col_hp_steer` ($n_\theta \le 2$, zero top coefficient, scaling with the magnitude of $u_{hp}$) apply here too.

`mean_error_to_resolve`
--------------------------------------------------------------------------------

```python
Error_Indicator.mean_error_to_resolve(self) -> float
```

Returns `self.error_to_resolve / max(nmarked, 1)`, where `nmarked` is the number of columns plus the number of cells with `do_ref = True`. It must be called after an `error_*` method (or `from_file`). It returns 0 when nothing is marked. No side effects.

This is the "error attempted to resolve" (EA2R) of Section 3.3: the average indicator over the marked elements. The spatial-angular steering criterion (Eq. 16) refines spatially with probability $p_{\vec{x}} = \bar{\eta}_{\vec{x}} / (\bar{\eta}_{\vec{x}} + \bar{\eta}_{\vec{s}})$. In `experiments/two-strong-scatterers/experiment.py` two indicators are built, one with `ref_kind = "spt"` (`error_spatial_jump`) and one with `ref_kind = "ang"` (`error_angular_jump`), so each average contains only spatial or only angular indicators:

```python
spt_EA2R = err_ind_spt.mean_error_to_resolve()
ang_EA2R = err_ind_ang.mean_error_to_resolve()
if rng.random() < spt_EA2R / (spt_EA2R + ang_EA2R):
    mesh = err_ind_spt.ref_by_ind()
else:
    mesh = err_ind_ang.ref_by_ind()
```

For a single indicator with `ref_kind = "all"`, the average mixes column and cell indicators.

`ref_by_ind`
--------------------------------------------------------------------------------

```python
Error_Indicator.ref_by_ind(self) -> Mesh
```

Returns a new `dg.mesh.Mesh`, refined according to the marks in `self.cols`. `self`, `self.proj` and `self.proj.mesh` are not modified; the refinements are applied to `copy.deepcopy(self.proj.mesh)`.

For each leaf column of the original mesh, in sorted key order, and only if that column key still exists in the copy:

1. if `ref_kind` is `"ang"` or `"all"`: for each leaf cell of the original column that still exists in the copy and has `do_ref = True`, call `mesh.ref_cell(col_key, cell_key, form = cell.ref_form)`;
2. if `ref_kind` is `"spt"` or `"all"`: if the column has `do_ref = True`, call `mesh.ref_col(col_key, kind = "spt", form = col.ref_form)`.

`Mesh.ref_cell` and `Mesh.ref_col` also refine neighbors to keep the mesh 1-irregular (for *h*) and to keep the even-degree criterion (for *p*). An element that such a cascade has already *h*-refined no longer has its key in the copy and is skipped, including any *p*-refinement it was marked for. An element that a cascade has already *p*-refined is still refined for its own mark. The function does no MPI communication; the experiment drivers call it on the root rank and broadcast the result.

`to_file`
--------------------------------------------------------------------------------

```python
Error_Indicator.to_file(self, file_path: str = "err_ind.json", **kwargs) -> None
```

Writes the indicator to `file_path` as JSON (overwriting it). It must be called after an `error_*` method. Keyword arguments and defaults:

- `write_projection` (`True`) : also write `self.proj` with `Projection.to_file(projection_file_path, **kwargs)`, a raw float64 vector written by `numpy.ndarray.tofile` (no header).
- `projection_file_path` (`"projection.npy"`) : where the projection is written.
- `write_mesh` (`True`) : passed on to `Projection.to_file`, which then writes the mesh with `Mesh.to_file(mesh_file_path)`. The mesh is written only when `write_projection` is also `True`.
- `mesh_file_path` (`"mesh.json"`) : where the mesh is written.

The experiment drivers call `to_file(path, write_projection = False, write_mesh = False)`, because they write the solution and mesh separately (`uh.npy`, `mesh.json`).

JSON format (keys of `"cols"` and `"cells"` are the integer column and cell keys written as strings, as `json` requires; only leaf elements of `self.proj.mesh` are written):

```
{
  "ref_kind": "all",              # self.ref_kind
  "ref_form": "hp",               # self.ref_form
  "ang_ref_tol": 0.5,             # self.ang_ref_tol
  "spt_ref_tol": 0.5,             # self.spt_ref_tol
  "error": 0.907,                 # self.error
  "error_to_resolve": 1.54,       # self.error_to_resolve
  "col_max_error": 0.907,         # self.col_max_error
  "cell_max_error": 0.637,        # self.cell_max_error
  "cols": {
    "0": {                        # column key
      "do_ref": true,
      "ref_form": "p",            # "h", "p", or "" if not marked
      "error": 0.907,
      "cells": {
        "1": {"do_ref": true,  "ref_form": "p", "error": 0.637},
        "2": {"do_ref": false, "ref_form": "",  "error": 0.270}
      }
    }
  }
}
```

(The `#` comments are explanatory and are not part of the file.) The tolerances written are the values given to the constructor, so the default tolerance is written as `1.7976931348623157e+308`.

`from_file`
--------------------------------------------------------------------------------

```python
from_file(mesh_file_path: str = "mesh.json",
          projection_file_path: str = "projection.npy",
          err_ind_file_path: str = "err_ind.json") -> Error_Indicator
```

Module-level function (not a method), exported by `__init__.py`. It reads the solution with `dg.projection.from_file(mesh_file_path, projection_file_path)`, reads the JSON written by `to_file`, builds `Error_Indicator(proj, ref_kind, ref_form, [ang_ref_tol, spt_ref_tol])`, sets `error`, `error_to_resolve`, `col_max_error` and `cell_max_error`, and rebuilds `cols` with integer keys from the stored `error`, `ref_form` and `do_ref` values. Nothing is recomputed. Every key in the JSON must be a leaf column or cell of the loaded mesh (otherwise `KeyError` or `AssertionError`), so the mesh and projection must be the ones the indicator was computed on. Example from `experiments/*/generate_experiment_visualizations.py`:

```python
from amr.error_indicator import from_file as error_indicator_from_file

err_ind = error_indicator_from_file(mesh_file_path, uh_file_path,
                                    err_ind_ang_jmp_file_path)
```

Source Files
--------------------------------------------------------------------------------

The source files contained in the `error_indicator` subdirectory are:

- `__init__.py` : Exports `Error_Indicator`, `from_file`, and the subpackages `error` and `error_indicator_column`.
- `cell_hp_steer.py` : Given an angular *hp*-element (`cell`) marked for angular refinement in an `Error_Indicator`, calculate the *hp*-steering criterion of Eq. (18b) (i.e., whether to mark it for angular *h*- or *p*-refinement).
- `class_Error_Indicator.py` : The class implementing the error indicator.
- `col_hp_steer.py` : Given a spatial *hp*-element (`column`, or `col`) marked for spatial refinement in an `Error_Indicator`, calculate the *hp*-steering criterion of SI Eq. (6b) (i.e., whether to mark it for spatial *h*- or *p*-refinement).
- `from_file.py` : Read an `Error_Indicator` from a mesh file, a projection file, and an error indicator JSON file.
- `mean_error_to_resolve.py` : The average error indicator over the elements marked for refinement in an `Error_Indicator`, used by the spatial-angular steering criterion (Eq. 16).
- `ref_by_ind.py` : Return a copy of the mesh refined according to an `Error_Indicator`.
- `to_file.py` : Write an `Error_Indicator` to a JSON file (and optionally its projection and mesh).

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `error_indicator` subdirectory is

```
error_indicator
|
|--- error : Contains source code for calculating various kinds of error for use in the error indicator (analytic, angular jump, spatial jump, high-resolution, and random), each of which also marks elements for refinement.
|--- error_indicator_column : Contains source code implementing the Error_Indicator_Column class, and the error_indicator_cell subdirectory implementing the Error_Indicator_Cell class.
```
