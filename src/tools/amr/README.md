`amr`
================================================================================

This subdirectory contains tools for visualizing error indicators (`amr.error_indicator.Error_Indicator`) and the refinement thresholds that decide which elements are marked for refinement.

How It Fits Into the Whole
--------------------------------------------------------------------------------

In the refinement loop (Section 3.2 of the publication) an `Error_Indicator` is built from the numerical solution, one of its `error_*` methods computes a spatial indicator $\eta_{\vec{x}}$ for each column $C$ and an angular indicator $\eta_{\vec{s}}$ for each spatial-angular element $(C, K)$ (Eqs. 13-14), and elements whose indicator reaches the threshold of Eq. 15 are marked. `experiments/*/experiment.py` writes the indicator to file (`Error_Indicator.to_file`); `experiments/*/generate_experiment_visualizations.py` reads it back with `amr.error_indicator.from_file` and calls:

- `plot_error_indicator(..., kind = "ang")` for the analytic and high-resolution indicators (`err_ind_anl.png`, `err_ind_hr.png`);
- `plot_refinement_thresholds(..., kind = "ang")` for the angular-jump indicator (`err_ind_ang_jmp.png`) and `plot_refinement_thresholds(..., kind = "spt")` for the spatial-jump indicator (`err_ind_spt_jmp.png`, Two Strong Scatterers).

The functions read these attributes of the `Error_Indicator` (set by the `error_*` methods or by `from_file`):

- `proj` (`dg.projection.Projection`) : the projection the indicator was computed from; its mesh gives `Ls`, the leaf columns and the leaf cells.
- `error` (`float`) : the indicator for the whole mesh.
- `error_to_resolve` (`float`) : the sum of the indicators of the marked elements.
- `col_max_error`, `cell_max_error` (`float`) : $\max_C \eta_{\vec{x}}(C)$ and $\max_{(C, K)} \eta_{\vec{s}}(C, K)$.
- `spt_ref_tol`, `ang_ref_tol` (`float`) : $\tau_{\vec{x}}$ and $\tau_{\vec{s}}$ of Eq. 15 (passed to `Error_Indicator` as `ref_tol = [ang_ref_tol, spt_ref_tol]`, default `consts.INF`).
- `cols[col_key].error`, `cols[col_key].cells[cell_key].error` (`float`) : the indicator of each column and of each (column, cell) pair.

They import `consts` (`PI`) and `amr.error_indicator`. They do not modify the indicator and do not use MPI.

Package Initialization
--------------------------------------------------------------------------------

`__init__.py` defines `src_dir` (`str`, the normalized path of the `src` directory), appends it to `sys.path` if absent, and exports:

- `plot_error_indicator` (`plot_error_indicator.py`)
- `plot_refinement_thresholds` (`plot_refinement_thresholds.py`)

`import tools` does not import `tools.amr`; import it explicitly (`from tools.amr import plot_error_indicator`).

Source Files
--------------------------------------------------------------------------------

The source files contained in the `amr` subdirectory are:

- `plot_error_indicator` : Plots the error indicator as a color plot which shows the value of the error on each spatial and each angular *hp*-element.
- `plot_refinement_thresholds` : Plots the error indicator in relation to the refinement threshold, which illustrates which elements are marked to be refined.

Both functions draw the mesh of `err_ind.proj` as in `tools/dg/mesh`: each leaf column is a rectangle, and its angular mesh is a disk of radius $\min(\Delta x, \Delta y)/2$ centred in the rectangle with one wedge per leaf cell from $\theta_0$ to $\theta_1$ (counterclockwise from $+x$). Axes are labelled `x` and `y`, with limits $[0, L_x] \times [0, L_y]$ unless `lims = [[x_min, x_max], [y_min, y_max]]` is given. The title has two lines, `Total Error: <error>` and `Error to Resolve: <error_to_resolve>`, in the format `{:4.4E}`. If `file_path` is given the figure is resized to 6.5 in $\times$ $6.5 L_y / L_x$ in, saved at 300 dpi and closed. Both return `[fig, ax]`. The keyword `name` (default `""`) is accepted but not used.

### `plot_error_indicator(err_ind, file_path = None, **kwargs)` (`plot_error_indicator.py`)

Colours each column by its spatial indicator and each wedge by its angular indicator.

- `err_ind` (`Error_Indicator`) : the indicator, with the attributes listed above.
- `file_path` (`str` or `None`, default `None`) : output image path.
- `lims` (`list`, default `[[], []]`) : plot ranges.
- `cmap` (`str`, default `"Reds"`) : `matplotlib` colormap.
- `kind` (`str`, default `"all"`) : `"spt"`, `"ang"` or `"all"`. The code gives the rectangles a face colour only for `"spt"` and `"all"`, and the wedges only for `"ang"` and `"all"`, setting the others to `"none"`.

Rectangles and wedges are two `PatchCollection`s that share the colormap and the colour limits `[0, max(col_max_error, cell_max_error)]`; the colour bar belongs to the wedges. Because each collection is given its values with `set_array`, `matplotlib` maps every face to the colormap, including the faces set to `"none"`: with `matplotlib` 3.9.2, a `kind = "ang"` plot still fills the rectangles with the column indicator colours. The wedges are drawn on top of the rectangles.

```python
from amr.error_indicator import from_file
from tools.amr import plot_error_indicator

err_ind = from_file("mesh.json", "uh.npy", "err_ind_hr.json")
plot_error_indicator(err_ind, "err_ind_hr.png", kind = "ang")
```

### `plot_refinement_thresholds(err_ind, file_path = None, **kwargs)` (`plot_refinement_thresholds.py`)

Shows each indicator relative to the refinement threshold $T = \tau \max \eta$ of Eq. 15, using sizes instead of colours. There is no colour bar.

- `err_ind` (`Error_Indicator`) : the indicator, with the attributes listed above.
- `file_path` (`str` or `None`, default `None`) : output image path.
- `lims` (`list`, default `[[], []]`) : plot ranges.
- `kind` (`str`, default `"all"`) : `"spt"` draws the spatial markers, `"ang"` the angular markers, `"all"` both.

All column rectangles and cell wedges are outlined in black. Then:

- Spatial (`kind` `"spt"` or `"all"`), in each column: a dashed black rectangle with sides $\tau_{\vec{x}} \Delta x \times \tau_{\vec{x}} \Delta y$ and a solid red rectangle with sides $r \Delta x \times r \Delta y$, $r = \eta_{\vec{x}}(C) / \max_C \eta_{\vec{x}}$, both centred in the column.
- Angular (`kind` `"ang"` or `"all"`), in each cell: a dashed black arc and a solid red arc from $\theta_0$ to $\theta_1$, centred at the column centre, with radii $\tau_{\vec{s}}$ and $\eta_{\vec{s}}(C, K) / \max \eta_{\vec{s}}$ times the disk radius $\min(\Delta x, \Delta y)/2$.

The red marker lies on or outside the dashed one exactly when $\eta \ge \tau \max \eta$, which is the test the `amr.error_indicator.error_*` methods use to mark an element (the text of Section 3.2 says "exceeds" the threshold). The element with the largest indicator has a red marker equal to the element boundary.

The markers divide by `col_max_error` and `cell_max_error`, so these must be nonzero for the kinds drawn, and $\tau$ must be finite: an indicator built without a tolerance for one kind has $\tau = $ `consts.INF` there, so call the function with the matching `kind` (as the experiment scripts do).

```python
from amr.error_indicator import from_file
from tools.amr import plot_refinement_thresholds

err_ind = from_file("mesh.json", "uh.npy", "err_ind_ang_jmp.json")
plot_refinement_thresholds(err_ind, "err_ind_ang_jmp.png", kind = "ang")
```
