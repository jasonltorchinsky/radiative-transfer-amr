# Standard Library Imports
import copy

# Third-Party Library Imports

# Local Library Imports
from dg.mesh import Mesh

# Relative Imports
from ..error_indicator.error_indicator_column import Error_Indicator_Column
from ..error_indicator.error_indicator_column.error_indicator_cell import Error_Indicator_Cell

def ref_by_ind(self) -> Mesh:

    mesh: Mesh = copy.deepcopy(self.proj.mesh) ## Copy the mesh to not mess up
    ## the Error_Indicator
    
    col_items: list = sorted(self.proj.mesh.cols.items())
    for col_key, col in col_items:
        assert(col.is_lf)

        if col_key in mesh.cols.keys(): # A column may have already been refined
            # in h to handle 1-irregularity. In that case, skip its p-refinement

            ## First handle cell refinements
            if self.ref_kind in ["ang", "all"]:
                cell_items: list = sorted(col.cells.items())
                for cell_key, cell in cell_items:
                    assert(cell.is_lf)
                    
                    if cell_key in mesh.cols[col_key].cells.keys(): # Avoid trying to refine cells
                        # that have been refined from 1-irregularity
                        err_ind_cell: Error_Indicator_Cell = self.cols[col_key].cells[cell_key]

                        if err_ind_cell.do_ref:
                            ref_form: str = err_ind_cell.ref_form
                            mesh.ref_cell(col_key, cell_key, form = ref_form)

            if self.ref_kind in ["spt", "all"]:    
                err_ind_col: Error_Indicator_Column = self.cols[col_key]
                if err_ind_col.do_ref:
                    ref_form: str = err_ind_col.ref_form

                    mesh.ref_col(col_key, kind = "spt", form = ref_form)
                    
    return mesh
