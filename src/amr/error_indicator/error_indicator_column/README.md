`error_indicator_column`
================================================================================

This subdirectory contains the source code regarding spatial elements (columns, or cols) in an error indicator.

How It Fits Into the Package
--------------------------------------------------------------------------------

An `Error_Indicator` stores one `Error_Indicator_Column` per leaf column of its mesh in the dict `Error_Indicator.cols`, keyed by the integer column key of `dg.mesh.Mesh.cols`. Each `Error_Indicator_Column` holds the column's spatial error indicator, its refinement mark, and the indicators of the column's leaf cells (`Error_Indicator_Cell`, defined in the `error_indicator_cell` subdirectory).

The objects are created by the `error_*` methods in `../error` and by `amr.error_indicator.from_file`. Their `do_ref` and `ref_form` fields are set by the marking block at the end of each `error_*` method and read by `Error_Indicator.ref_by_ind` (spatial refinement of the column with `Mesh.ref_col(col_key, kind = "spt", form = ref_form)`), `Error_Indicator.mean_error_to_resolve`, `Error_Indicator.to_file`, and the plotting tools in `src/tools/amr`.

Exported Names
--------------------------------------------------------------------------------

`__init__.py` exports `Error_Indicator_Column` and the subpackage `error_indicator_cell`.

`Error_Indicator_Column`
--------------------------------------------------------------------------------

```python
Error_Indicator_Column(error: float, cells: dict, ref_form: str = "", do_ref: bool = False)
```

A plain data container; it has no methods besides the constructor.

Arguments and attributes:

- `error` (`float`) : the column's error indicator. Its meaning depends on the `error_*` method that created it (e.g. the spatial-jump indicator $\eta_{\vec{x}}^{C}$ of Eq. 13, or $\sqrt{\sum_{K \in C} e_K / \Vert u \Vert^2}$ for the analytic error); see `../error/README.md`.
- `cells` (`dict[int, Error_Indicator_Cell]`) : the indicators of the column's leaf cells, keyed by the integer cell keys of `Column.cells`.
- `ref_form` (`str`, default `""`) : the form of spatial refinement, `"h"` or `"p"`, when the column is marked; `""` when it is not.
- `do_ref` (`bool`, default `False`) : whether the column is marked for spatial refinement. It is used only when the indicator's `ref_kind` is `"spt"` or `"all"`.

Example:

```python
from amr.error_indicator.error_indicator_column import Error_Indicator_Column
from amr.error_indicator.error_indicator_column.error_indicator_cell import Error_Indicator_Cell

col_ind = Error_Indicator_Column(0.1, {0: Error_Indicator_Cell(0.05)})
col_ind.do_ref, col_ind.ref_form = True, "h"
```

Source Files
--------------------------------------------------------------------------------

The source files contained in the `error_indicator_column` subdirectory are:

- `__init__.py` : Exports `Error_Indicator_Column` and the `error_indicator_cell` subpackage.
- `class_Error_Indicator_Column.py` : The class implementing spatial elements (columns, or cols) in an error indicator.

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `error_indicator_column` subdirectory is

```
error_indicator_column
|
|--- error_indicator_cell : Contains source code implementing the Error_Indicator_Cell class.
```
