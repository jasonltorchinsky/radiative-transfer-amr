`column`
================================================================================

This subdirectory contains the source code regarding spatial *hp*-elements (`columns`, or `cols`) in an *hp*-mesh.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `column` subdirectory are:

- `add_cell` : Adds an angular *hp*-element (`cell`) to the angular *hp*-mesh associated with the spatial *hp*-element (`column`, or `col`).
- `calc_key` : Calculates the key for the spatial *hp*-element (`column`, or `col`).
- `class_Column` : The class implementing spatial elements (`columns`, or `cols`) in an *hp*-mesh.
- `del_cell` : Deletes an angular *hp*-element (`cell`) from the angular *hp*-mesh associated with the spatial *hp*-element (`column`, or `col`).
- `nhbr_cell` : Returns the keys for the neighboring angular *hp*-elements (`cells`) in the angular *hp*-mesh associated with the spatial *hp*-element (`column`, or `col`).

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `column` subdirectory is

```
column
|
|--- cell : Contains source code implementing the Cell class.
```