`test_cell`
================================================================================

This subdirectory contains the unit tests for angular *hp*-elements (`cells`), the `Cell` class of `src/dg/mesh/column/cell`. A `Cell` is one arc $K = [\theta_0, \theta_1] \subset [0, 2\pi]$ of the angular mesh of a column. It stores `pos = [th0, th1]`, its index `idx` and level `lv` (from which its key $2^{lv} - 1 + idx$ is computed), whether it is a leaf, its number of Legendre-Gauss nodes `ndofs = [nth]`, the angular `quadrant` it lies in (`None` if it spans more than one), and `nhbr_keys`, the keys of the neighboring cells in the same column. Each `Column` holds its cells in a dictionary keyed by cell key.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `test_cell` subdirectory are:

- `test_Cell.py` : Unit test for the `Cell` class.

`test_Cell.py`
--------------------------------------------------------------------------------

- `test_Cell(tmp_path)` : builds three cells with `Cell(pos, idx, lv, is_lf, ndofs, quadrant, nhbr_keys)` = `([0., 2. * consts.PI], 0, 0, True, [3], None, [None, None])`, an identical copy, and one with `pos = [0., consts.PI]`. It asserts that `Cell.__eq__` returns `True` for the identical cells and `False` for the cell with the different `pos`, and writes `cell.__str__()` to `cell_str.txt` in `tmp_path`. The text output is not checked. (`Cell.__eq__` compares every attribute but not the column the cell belongs to.)
