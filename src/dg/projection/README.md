`projection`
================================================================================

This subdirectory contains the source code regarding a projection of a function to an *hp*-mesh.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `projection` subdirectory are:

- `cell_intg_xy` : Integrates an angular-spatial *hp*-element (`cell`) in space.
- `class_Projection` : Class implementing the projection of a function to an *hp*-mesh.
- `col_intg_th` : Integrates the numerical solution on an angular *hp*-mesh associated with a given spatial *hp*-element (`column`, or `col`) in angle.
- `from_file` : Reads a `Projection` from a file.
- `from_vector` : Converts a `numpy` `ndarray` to a `Projection`.
- `get_f2f_matrix` : Returns a matrix for mapping between faces of neighboring spatial-angular *hp*-elements.
- `intg_th` : Integrates the projection in angle.
- `push_pull` : Maps between the intervals $\left[ x_0,\ x_f \right]$ and $\left[ -1,\ 1 \right]$.
- `to_file` : Writes a `Projection` to a file.
- `to_vector` : Converts a `Projection` to a `numpy` `ndarray`.

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `projection` subdirectory is

```
projection
|
|--- projection_column : Contains source code implementing the Projection_Column class.
```