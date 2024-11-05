`error_indicator`
================================================================================

This subdirectory contains the source code regarding the error indicator.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `error_indicator` subdirectory are:

- `cell_hp_steer` : Given an angular *hp*-element (`cell`) marked for angular refinement in an `Error_Indicator`, calculate the *hp*-steering criterion (i.e., whether to mark it for angular *h*- or *p*-refinement).
- `class_Error_Indicator` : The class implementing the error indicator.
- `col_hp_steer` : Given an spatial *hp*-element (`column`, or `col`) marked for spatial refinement in an `Error_Indicator`, calculate the *hp*-steering criterion (i.e., whether to mark it for spatial *h*- or *p*-refinement).
- `from_file` : Read an `Error_Indicator` from a file.
- `ref_by_ind` : Refine a mesh according to an Error_Indicator.
- `to_file` : Write an `Error_Indicator` to file.

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `error_indicator` subdirectory is

```
error_indicator
|
|--- error : Contains source code for calulating various kinds of error for use in the error indicator, including, e.g., analytic error and random error.
|--- error_indicator_column : Contains source code implementing the Error_Indicator_Column class.
```