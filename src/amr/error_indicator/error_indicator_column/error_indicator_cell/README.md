`error_indicator_cell`
================================================================================

This subdirectory contains the source code regarding angular elements (cells) in a spatial element of an error indicator.

How It Fits Into the Package
--------------------------------------------------------------------------------

Each `Error_Indicator_Column` stores one `Error_Indicator_Cell` per leaf cell of its column in the dict `Error_Indicator_Column.cells`, keyed by the integer cell key of `dg.mesh.column.Column.cells`. A cell indicator therefore belongs to one spatial-angular element $(C, K)$. The cell at column key `c` and cell key `k` of an `Error_Indicator` named `err_ind` is `err_ind.cols[c].cells[k]`.

The objects are created by the `error_*` methods in `../../error` and by `amr.error_indicator.from_file`. Their `do_ref` and `ref_form` fields are set by the marking block at the end of each `error_*` method and read by `Error_Indicator.ref_by_ind` (angular refinement with `Mesh.ref_cell(col_key, cell_key, form = ref_form)`), `Error_Indicator.mean_error_to_resolve`, `Error_Indicator.to_file`, and the plotting tools in `src/tools/amr`.

Exported Names
--------------------------------------------------------------------------------

`__init__.py` exports `Error_Indicator_Cell`.

`Error_Indicator_Cell`
--------------------------------------------------------------------------------

```python
Error_Indicator_Cell(error: float, ref_form: str = "", do_ref: bool = False)
```

A plain data container; it has no methods besides the constructor.

Arguments and attributes:

- `error` (`float`) : the error indicator of the spatial-angular element. Its meaning depends on the `error_*` method that created it (e.g. the angular-jump indicator $\eta_{\vec{s}}^{(C,K)}$ of Eq. 14); see `../../error/README.md`. `error_spatial_jump` stores the placeholder `-consts.INF` here because it computes column indicators only.
- `ref_form` (`str`, default `""`) : the form of angular refinement, `"h"` or `"p"`, when the cell is marked; `""` when it is not.
- `do_ref` (`bool`, default `False`) : whether the cell is marked for angular refinement. It is used only when the indicator's `ref_kind` is `"ang"` or `"all"`.

Example:

```python
from amr.error_indicator.error_indicator_column.error_indicator_cell import Error_Indicator_Cell

cell_ind = Error_Indicator_Cell(0.05, ref_form = "p", do_ref = True)
```

Source Files
--------------------------------------------------------------------------------

The source files contained in the `error_indicator_cell` subdirectory are:

- `__init__.py` : Exports `Error_Indicator_Cell`.
- `class_Error_Indicator_Cell.py` : The class implementing angular elements (cells) in a spatial element of an error indicator.
