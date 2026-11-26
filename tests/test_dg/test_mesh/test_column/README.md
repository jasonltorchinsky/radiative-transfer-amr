`test_column`
================================================================================

This subdirectory contains the unit tests for spatial *hp*-elements (`columns`, or `cols`), the `Column` class of `src/dg/mesh/column`. A `Column` is one rectangle $C$ of the spatial mesh. It stores its corners `pos = [x0, y0, x1, y1]`, its index and refinement level, its key, whether it is a leaf, its numbers of LGL nodes `ndofs = [nx, ny]`, a dictionary `cells` of the angular elements (`Cell`s) over it, and `nhbr_keys`, the keys of up to two neighboring columns across each of its four faces (index 0 is the right face, then counterclockwise: top, left, bottom). `Mesh` creates, refines and deletes columns; `Projection` and `rt.Problem` loop over them.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `test_column` subdirectory are:

- `test_Column.py` : Unit test for the `Column` class.

`test_Column.py`
--------------------------------------------------------------------------------

- `test_Column(tmp_path)` : builds two `Cell`s that differ only in `pos` ($[0, 2\pi]$ and $[0, \pi]$) and three `Column`s with the same arguments (`pos = [[0., 1.], [-0.5, 0.5]]`, `idx = [0, 0]`, `lv = 0`, `is_lf = True`, `ndofs = [3, 4]`, no neighbors); two of them hold the first cell and the third holds the second cell. It asserts that `Column.__eq__` returns `True` for the two equal columns and `False` for the column with the different cell, and writes `col.__str__()` to `col_str.txt` in `tmp_path`.

  The test passes `Column(pos, idx, lv, is_lf, ndofs, cells, nhbr_keys)` positionally. Its `pos` is a nested list, not the flat `[x0, y0, x1, y1]` that `Mesh` uses; `Column.__eq__` and `Column.__str__` do not unpack `pos`, so the test is not affected. The text output is not checked.

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `test_column` subdirectory is

```
test_column
|
|--- test_cell : Contains unit tests for the Cell class.
```
