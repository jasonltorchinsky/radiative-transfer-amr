# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports
import consts
from dg.projection import Projection

# Relative Imports


class Error_Indicator():
    def __init__(self, proj: Projection, ref_kind: str = "all", 
                 ref_form: str = "hp", ref_tol: list = [consts.INF, consts.INF]):
        self.proj: Projection = proj
        self.ref_kind: str = ref_kind
        self.ref_form: str = ref_form
        [self.ang_ref_tol, self.spt_ref_tol] = ref_tol # [ang_ref_tol, spt_ref_tol]

    from .cell_hp_steer import cell_hp_steer
    from .col_hp_steer import col_hp_steer

    from .error import (error_analytic, error_cell_jump, error_column_jump,
                        error_high_resolution, error_random)
    from .ref_by_ind import ref_by_ind

    from .to_file import to_file
