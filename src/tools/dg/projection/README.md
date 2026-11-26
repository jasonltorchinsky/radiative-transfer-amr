`projection`
================================================================================

This subdirectory contains tools for visualizing a numerical projection on an *hp*-mesh (`dg.projection.Projection`): integrals of $u(x, y, \theta)$ over one or two of its dimensions, and its mean on each spatial-angular *hp*-element.

How It Fits Into the Whole
--------------------------------------------------------------------------------

A `Projection` stores the nodal values of a function on each leaf (column, cell) pair as `proj.cols[col_key].cells[cell_key].vals`, an array of shape `[nx, ny, nth]` indexed `[i, j, a]`, at the nodes of `dg.quadrature.quad_xyth`: Legendre-Gauss-Lobatto (LGL) nodes in $x$ and $y$, Legendre-Gauss (LG) nodes in $\theta$, mapped from $[-1, 1]$ to the element with `dg.projection.push_forward`. The functions here integrate those values with the matching quadrature weights $w$ and plot the result with `matplotlib`.

Callers:

- `experiments/*/generate_experiment_visualizations.py` : all five functions on the numerical solution `uh` (`uh_th.png`, `uh_xy.png`, `uh_xth.png`, `uh_yth.png`, `uh_xyth.png`, with `scale = "pos"` except `plot_th`) and on the high-resolution solution `uh_hr` (`uh_hr_*.png`).
- `experiments/*/generate_problem_visualizations.py` : `plot_xy` of the extinction coefficient $\kappa$ and scattering coefficient $\sigma$ (`kappa.png`, `sigma.png`, with `show_mesh = False, cmap = "gray"`; the scripts project $\kappa / (2\pi)$ and $\sigma / (2\pi)$ so that the angular integral returns $\kappa$ and $\sigma$), and for the Manufactured Solution all five functions on the exact solution (`u_*.png`).
- `experiments/test-rt/experiment.py` : all five functions on the test solution `uh`.

The angular integral and the $x$-integral correspond to the "angular integral" and "horizontal integral" panels of Figs. 5 and 8 of the publication. These functions import `consts` (`PI`, `INF`), `dg.projection` (`Projection`, `push_forward`) and `dg.quadrature` (`quad_xyth`). They do not modify the projection and do not use MPI.

Package Initialization
--------------------------------------------------------------------------------

`__init__.py` defines `src_dir` (`str`) and appends it to `sys.path` if absent (the computed path is `src/src`, which does not exist, so `src` must already be on `sys.path`), and exports `plot_th`, `plot_xy`, `plot_xth`, `plot_yth` and `plot_xyth`.

Common Arguments and Behaviour
--------------------------------------------------------------------------------

All five functions have the signature `plot_*(proj, file_path = None, **kwargs)` and return `[fig, ax]` (`matplotlib.figure.Figure`, `matplotlib.axes.Axes`).

- `proj` (`dg.projection.Projection`) : the function to plot; every column and cell must be a leaf. The functions are written for projections with an angular dimension (`mesh.has_th = True`); on a mesh without one, each column has a single dummy cell with `pos = [0, 0]`, so the $\theta$-integrals have zero length.
- `file_path` (`str` or `None`, default `None`) : if given, the figure is resized to a width of 6.5 in and a height proportional to the plotted domain, saved at 300 dpi and closed.
- `lims` (`list` of two `list`, default `[[], []]`) : `[[h_min, h_max], [v_min, v_max]]` for the horizontal and vertical axes; an empty list means the full range ($[0, L_x]$, $[0, L_y]$ or $[0, 2\pi]$).
- `cmap` (`str`, default `"hot"`) : name of a `matplotlib` colormap.
- `scale` (`str`, default `"normal"`) : colour limits. `"normal"` uses the minimum and maximum of the plotted values; `"diff"` uses $[-m, m]$ with $m = \max(|v_{min}|, |v_{max}|)$; `"pos"` sets the lower limit to 0 and calls `cmap.set_under("blue")`, so negative values are blue. Any other value behaves like `"normal"`.
- `show_mesh` (`bool`, default `True`; not in `plot_xyth`) : draw the element boundaries as black rectangles.

The colour bar always uses `extend = "min"`. Plots that use `pcolormesh(..., shading = "gouraud")` interpolate between nodes. The LGL nodes in $x$ and $y$ include the element endpoints, so spatial elements are fully coloured; the LG nodes in $\theta$ do not, so a thin uncoloured strip appears at each angular element boundary in `plot_xth` and `plot_yth` (as noted in the caption of Fig. 5 of the publication).

Source Files
--------------------------------------------------------------------------------

The source files contained in the `projection` subdirectory are:

- `plot_th` : Visualizes the spatially-integrated numerical projection.
- `plot_xth` : Visualizes the $y$-integrated numerical projection.
- `plot_xy` : Visualizes the angularly-integrated numerical projection.
- `plot_xyth` : Visualizes the numerical projection averaged across each spatial-angular *hp*-element.
- `plot_yth` : Visualizes the $x$-integrated numerical projection.

### `plot_th(proj, file_path = None, **kwargs)` (`plot_th.py`)

Plots, in each column $C$, the angular distribution of the spatial integral

$U_C(\theta) = \int_{y_0}^{y_1} \int_{x_0}^{x_1} u(x, y, \theta) \, dx \, dy \approx \frac{\Delta x \Delta y}{4} \sum_{i, j} w_i w_j u_{ija}$

at the LG nodes $\theta_a$ of every cell of the column (concatenated over cells and sorted by $\theta$).

- Main axes: `x` and `y`, limits from `lims`; the column boundaries are always drawn as black rectangles (`show_mesh`, `cmap` and `scale` are accepted but not used).
- In each column, an inset polar axes (`ax.inset_axes` with `projection = "polar"`) covers the column's rectangle and shows $r = U_C(\theta)$ as a black curve, with $\theta$ counterclockwise from $+x$. The radial range is $[r_{min}, r_{max}]$, the minimum and maximum of $U_C$ over all columns, so the centre of each inset is $r_{min}$, not 0, and all insets share one scale. Angular and radial tick labels are removed; radial grid circles are placed at $0.33 r_{max}$ and $0.67 r_{max}$.
- The inset position is given in axes-fraction coordinates `[x0/Lx, y0/Ly, dx/Lx, dy/Ly]`, so it matches the columns only for the default limits $[0, L_x] \times [0, L_y]$.
- Saved size: 6.5 in $\times$ $6.5 L_y / L_x$ in. There is no colour bar and no title.

### `plot_xy(proj, file_path = None, **kwargs)` (`plot_xy.py`)

Plots the angular integral

$U(x, y) = \int_0^{2\pi} u(x, y, \theta) \, d\theta \approx \sum_K \frac{\Delta\theta_K}{2} \sum_a w_a u_{ija}$

in each column, at the column's LGL nodes, with `pcolormesh` (Gouraud shading) on axes `x` (horizontal) and `y` (vertical), a colour bar, and the column boundaries if `show_mesh`. Saved size: 6.5 in $\times$ $6.5 L_y / L_x$ in. This is the "angular integral" panel of Figs. 5(a) and 8(a) of the publication.

```python
import numpy as np
from dg.mesh import Mesh
from dg.projection import Projection
from tools.dg.projection import plot_xy, plot_xth

mesh = Mesh(Ls = [2., 1.], pbcs = [False, False], ndofs = [3, 3, 4], has_th = True)
u = Projection(mesh, lambda x, y, th: np.exp(-x) * (1. + np.cos(th)))
plot_xy(u, "u_xy.png", scale = "pos")
plot_xy(u, "u_xy_nomesh.png", show_mesh = False, cmap = "gray")
```

### `plot_xth(proj, file_path = None, **kwargs)` (`plot_xth.py`)

For each (column, cell) pair, computes the integral over the column's own $y$-range,

$\int_{y_0}^{y_1} u(x, y, \theta) \, dy \approx \frac{\Delta y}{2} \sum_j w_j u_{ija}$ (shape `[nx, nth]`),

and draws it with `pcolormesh` (Gouraud shading) on the column's $x$-nodes and the cell's $\theta$-nodes. Axes: `x` horizontal ($[0, L_x]$), $\theta$ vertical ($[0, 2\pi]$, ticks at multiples of $\pi/4$ labelled `0.00π`, `0.25π`, ..., `2.00π`); `lims = [[x_min, x_max], [th_min, th_max]]`. With `show_mesh`, each (column, cell) pair is outlined as the rectangle $[x_0, x_1] \times [\theta_0, \theta_1]$. Saved size: 6.5 in $\times$ $6.5 \cdot 2\pi / L_x$ in.

The values are not summed over columns that share an $x$-range: each column is drawn at the same $(x, \theta)$ position, and the column drawn last (largest column key) is the one visible. The figure therefore shows the integral over $[0, L_y]$ only when the mesh has a single column in $y$; otherwise it shows the integral over one column's $y$-range. The colour limits are taken over all columns.

### `plot_yth(proj, file_path = None, **kwargs)` (`plot_yth.py`)

As `plot_xth` with $x$ and $y$ exchanged: for each (column, cell) pair, computes

$\int_{x_0}^{x_1} u(x, y, \theta) \, dx \approx \frac{\Delta x}{2} \sum_i w_i u_{ija}$ (shape `[ny, nth]`),

and draws it on the cell's $\theta$-nodes and the column's $y$-nodes. Axes: $\theta$ horizontal ($[0, 2\pi]$, ticks at multiples of $\pi/4$), `y` vertical ($[0, L_y]$); `lims = [[th_min, th_max], [y_min, y_max]]`. With `show_mesh`, each pair is outlined as $[\theta_0, \theta_1] \times [y_0, y_1]$. Saved size: 6.5 in $\times$ $6.5 L_y / (2\pi)$ in.

As in `plot_xth`, columns that share a $y$-range are drawn on top of each other rather than summed, so the figure shows the integral over $[0, L_x]$ only when the mesh has a single column in $x$. Figs. 5(b) and 8(b) of the publication are described as the "horizontal integral" of $u$, i.e., over the whole $x$-range.

### `plot_xyth(proj, file_path = None, **kwargs)` (`plot_xyth.py`)

Plots the mean of $u$ on each spatial-angular element $(C, K)$,

$\bar{u}_{(C,K)} = \frac{1}{\Delta x \Delta y \Delta\theta} \int_K \int_C u \, dx \, dy \, d\theta \approx \frac{1}{8} \sum_{i, j, a} w_i w_j w_a u_{ija}$,

as the colour of the cell's wedge in the column's disk (radius $\min(\Delta x, \Delta y)/2$, centred in the column, from $\theta_0$ to $\theta_1$ counterclockwise from $+x$). Column boundaries are always drawn as black rectangles (there is no `show_mesh` option; a `show_mesh` keyword is ignored). Axes `x` and `y`, colour bar for the wedge colours (limits from `scale`). Saved size: 6.5 in $\times$ $6.5 L_y / L_x$ in.

```python
from tools.dg.projection import plot_xyth

fig, ax = plot_xyth(u)  # figure left open for further changes
ax.set_title("Element means of u")
fig.savefig("u_xyth.png", dpi = 300)
```
