`dg`
================================================================================

This subdirectory contains tools for visualizing an *hp*-mesh and a numerical projection on an *hp*-mesh. It mirrors the solver package `dg` (`src/dg`): `tools.dg.mesh` plots `dg.mesh.Mesh` objects and `tools.dg.projection` plots `dg.projection.Projection` objects.

How It Fits Into the Whole
--------------------------------------------------------------------------------

The experiment scripts (`experiments/*/generate_*_visualizations.py`, `experiments/test-rt/experiment.py`) load a mesh and a projection (the numerical solution `uh`, a high-resolution solution `uh_hr`, an exact solution, or a coefficient such as $\kappa$ or $\sigma$) and pass them here. `tools.dg` imports `consts`, `dg.mesh`, `dg.projection` and `dg.quadrature`; nothing in `src` imports `tools.dg`. The shared conventions (keyword arguments `lims`, `cmap`, `scale`, `file_path`, the drawing of angular meshes as disks, return values) are described in `tools/README.md`.

Package Initialization
--------------------------------------------------------------------------------

`__init__.py` defines `src_dir` (`str`) and appends it to `sys.path` if absent; the path it computes is `src/src`, which does not exist, so `src` must already be on `sys.path`. It imports only the subpackage `projection`. `tools.dg.mesh` is not imported by `tools.dg` and must be imported explicitly:

```python
from tools.dg.mesh import plot_mesh, plot_nhbrs
from tools.dg.projection import plot_xy
```

This directory has no source files of its own; the functions are documented in `mesh/README.md` and `projection/README.md`.

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `dg` subdirectory is

```
dg
|
|--- mesh : Contains tools for visualizing an *hp*-mesh (plot_mesh) and the neighbors of a spatial or spatial-angular element (plot_nhbrs).
|--- projection : Contains tools for visualizing functions projected to an *hp*-mesh: spatial integral (plot_th), angular integral (plot_xy), y-integral (plot_xth), x-integral (plot_yth) and element means (plot_xyth).
```
