`mesh`
================================================================================

This subdirectory contains the source code regarding the *hp*-mesh.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `mesh` subdirectory are:

- `add_col` : Adds a spatial *hp*-element (`column` , or `col`) to the *hp*-mesh.
- `class_Mesh` : Source code for the *hp*-mesh.
- `del_col` : Deletes a spatial *hp*-element (`column` , or `col`) from the *hp*-mesh.
- `from_file` : Reads an *hp*-mesh from a file.
- `get_ndof` : Count the number of degrees of freedom in the *hp*-mesh.
- `nhbr_cells_in_nhbr_col` : Returns the keys for neighboring angular *hp*-elements (`cells`) in a neighboring spatial *hp*-element (`column`, or `col`).
- `nhbr_cells_spt` : Returns the keys for all neighboring angular *hp*-elements (`cells`) in all neighboring spatial *hp*-elements (`columns`, or `cols`).
- `nhbr_cols` : Returns the keys for all neighboring spatial *hp*-elements (`columns`, or `cols`).
- `ref_cell` : Angularly refines a given angular *hp*-element (`cell`), maintaining all mesh irregularity criteria.
- `ref_col_ang` : Angularly refines the angular *hp*-mesh (the `cells`) associated with a given spatial *hp*-element (`column`, or `col`).
- `ref_col_spt` : Spatially refines a given spatial *hp*-element (`column`, or `col`).
- `ref_col` : Refines a given spatial *hp*-element (`column`, or `col`).
- `ref_mesh` : Refines all *hp*-elements in the *hp*-mesh.
- `remove_th` : Removes the angular dimension of the *hp*-mesh.
- `to_file`: Writes the *hp*-mesh to file.

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `mesh` subdirectory is

```
mesh
|
|--- mesh_column : Contains source code implementing the Column class.
```