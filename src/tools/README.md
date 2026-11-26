`tools`
================================================================================

This subdirectory contains source code for generating relevant plots with `matplotlib`, e.g., of the *hp*-mesh, numerical solutions on the *hp*-mesh, error indicators, and matrices. It is used only for visualization: nothing in `consts`, `utils`, `dg`, `rt` or `amr` imports `tools`.

How It Fits Into the Whole
--------------------------------------------------------------------------------

`tools` sits at the end of the pipeline. The solve and refinement loop (`dg.mesh.Mesh` -> `rt.Problem.solve` -> `amr.error_indicator.Error_Indicator` -> `ref_by_ind`) writes meshes, projections and error indicators to file; the plotting scripts in `experiments` read them back (`dg.projection.from_file`, `amr.error_indicator.from_file`) and pass them to the functions below:

- `experiments/*/generate_problem_visualizations.py` : `plot_xy` of the extinction and scattering coefficients (`kappa.png`, `sigma.png`) and, for the Manufactured Solution, all five projection plots of the exact solution (`u_th.png`, `u_xy.png`, `u_xth.png`, `u_yth.png`, `u_xyth.png`).
- `experiments/*/generate_experiment_visualizations.py` : for each trial and refinement step, `plot_mesh` (`mesh.png`, `mesh_hr.png`), the five projection plots of the numerical solution `uh` and of the high-resolution solution `uh_hr` (`uh_*.png`, `uh_hr_*.png`), `plot_error_indicator` (`err_ind_anl.png`, `err_ind_hr.png`) and `plot_refinement_thresholds` (`err_ind_ang_jmp.png`, `err_ind_spt_jmp.png`).
- `experiments/test-rt/experiment.py` : `plot_mesh`, the projection plots, and `plot_matrix` of the mass, scattering, convection and system matrices.

`tools` imports `consts` (for `PI` and `INF`), `dg.mesh`, `dg.projection` and `dg.quadrature` (to read the mesh and to integrate projections with the quadrature rule they were built on), and `amr.error_indicator` (type hints only). None of the functions uses MPI. The experiment scripts call them from a single process: `run.sh` starts the visualization scripts with `mpirun -n 1`, and `test-rt/experiment.py` calls them inside `if comm_rank == consts.COMM_ROOT`.

Package Initialization
--------------------------------------------------------------------------------

The packages under `tools` mirror the names of the top-level packages (`tools.amr`, `tools.dg`, `tools.rt`), so they are imported with the `tools.` prefix once `src` is on `sys.path`:

```python
from tools.dg.mesh import plot_mesh, plot_nhbrs
from tools.dg.projection import plot_th, plot_xy, plot_xth, plot_yth, plot_xyth
from tools.amr import plot_error_indicator, plot_refinement_thresholds
from tools.rt import plot_matrix
```

Do not put `src/tools` itself on `sys.path`, since `tools/dg`, `tools/amr` and `tools/rt` would then shadow the solver packages `dg`, `amr` and `rt`.

Each `__init__.py` defines `src_dir` (`str`) and appends it to `sys.path` if absent. Only `tools/amr/__init__.py` resolves to the `src` directory; the others (`tools`, `tools/dg`, `tools/dg/mesh`, `tools/dg/projection`, `tools/rt`) resolve to `src/src`, which does not exist, so importing them only works when `src` is already on `sys.path` (as the experiment scripts arrange).

`tools/__init__.py` imports only the subpackage `dg`, and `tools/dg/__init__.py` imports only `projection`. `import tools` therefore makes `tools.dg` and `tools.dg.projection` available, but `tools.dg.mesh`, `tools.amr` and `tools.rt` must be imported explicitly.

Conventions Shared by the Plotting Functions
--------------------------------------------------------------------------------

- **Inputs.** The functions read a `dg.mesh.Mesh`, a `dg.projection.Projection` or an `amr.error_indicator.Error_Indicator` and never modify it. They assert that every column and cell they visit is a leaf (`is_lf`). They loop over columns in increasing column key and over cells in increasing cell key.
- **Domain.** Space is $[0, L_x] \times [0, L_y]$ with `[Lx, Ly] = mesh.Ls`, and angle is $\theta \in [0, 2\pi]$. A column has `pos = [x0, y0, x1, y1]` and `ndofs = [nx, ny]`; a cell has `pos = [th0, th1]` (radians) and `ndofs = [nth]`; the values of a projection on a (column, cell) pair are `vals` of shape `[nx, ny, nth]` at the nodes of `dg.quadrature.quad_xyth` (Legendre-Gauss-Lobatto in $x$ and $y$, Legendre-Gauss in $\theta$), mapped to the element with `dg.projection.push_forward`.
- **Spatial-angular mesh drawing.** As in Fig. 1(c) and Fig. 9 of the publication, a spatial element is drawn as a rectangle, and its angular mesh is drawn as a disk centred at the rectangle's centre with radius $\min(\Delta x, \Delta y)/2$, in which each cell is a wedge from $\theta_0$ to $\theta_1$, measured counterclockwise from the $+x$ axis (the direction $\vec{s} = (\cos\theta, \sin\theta)$). The axes are not set to an equal aspect ratio, so the disks appear as ellipses when the axes box is not proportional to $L_x \times L_y$.
- **`lims`** (`list` of two `list`, default `[[], []]`) : plot ranges `[[h_min, h_max], [v_min, v_max]]` for the horizontal and vertical axes. An empty list selects the full range of that axis ($[0, L_x]$, $[0, L_y]$ or $[0, 2\pi]$).
- **`file_path`** : if given (truthy), the figure is resized to a width of 6.5 inches and a height in proportion to the plotted domain, laid out with `plt.tight_layout`, written with `plt.savefig(file_path, dpi = 300)` (the format follows the file extension), and closed. If `file_path` is `None`, the figure is left open. Except for `plot_matrix`, every function returns `[fig, ax]` in both cases (after saving, `fig` is closed).
- **Colour scales** (`dg.projection` plots) : `cmap` (`str`, default `"hot"`) is a `matplotlib` colormap name. `scale` (`str`, default `"normal"`) sets the colour limits: `"normal"` uses the minimum and maximum of the plotted values, `"diff"` uses $[-m, m]$ with $m = \max(|v_{min}|, |v_{max}|)$ (for differences), and `"pos"` sets the lower limit to 0 and draws values below 0 in blue. The colour bar always has `extend = "min"`.
- Unknown keyword arguments are accepted and ignored.

Function Summary
--------------------------------------------------------------------------------

| Function | Package | Input | Figure |
| --- | --- | --- | --- |
| `plot_mesh` | `tools.dg.mesh` | `Mesh` | Spatial-angular *hp*-mesh, elements coloured by number of nodes |
| `plot_nhbrs` | `tools.dg.mesh` | `Mesh`, keys | One element and its neighbours |
| `plot_th` | `tools.dg.projection` | `Projection` | $\iint_C u \, dx \, dy$ as a polar curve in each column |
| `plot_xy` | `tools.dg.projection` | `Projection` | $\int_0^{2\pi} u \, d\theta$ over $(x, y)$ |
| `plot_xth` | `tools.dg.projection` | `Projection` | $\int u \, dy$ over each column's $y$-range, over $(x, \theta)$ |
| `plot_yth` | `tools.dg.projection` | `Projection` | $\int u \, dx$ over each column's $x$-range, over $(\theta, y)$ |
| `plot_xyth` | `tools.dg.projection` | `Projection` | Mean of $u$ on each spatial-angular element, as coloured wedges |
| `plot_error_indicator` | `tools.amr` | `Error_Indicator` | Error indicator of each column and cell, as colours |
| `plot_refinement_thresholds` | `tools.amr` | `Error_Indicator` | Error indicator relative to the refinement threshold (Eq. 15) |
| `plot_matrix` | `tools.rt` | 2-D array | Sparsity pattern |

Each function is documented in the `README.md` of its subdirectory.

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `tools` subdirectory resembles that of the `src` directory, and is

```
tools
|
|--- amr : Contains tools for visualizing an error indicator and the refinement thresholds (plot_error_indicator, plot_refinement_thresholds).
|--- dg : Contains tools for visualizing an *hp*-mesh and a numerical projection on an *hp*-mesh.
|    |--- mesh : Plots an *hp*-mesh and the neighbors of an element (plot_mesh, plot_nhbrs).
|    |--- projection : Plots integrals and element means of a projection (plot_th, plot_xy, plot_xth, plot_yth, plot_xyth).
|--- rt : Contains tools for visualizing the sparsity pattern of matrices (plot_matrix).
```
