# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports

# Relative Imports
from .class_Projection import Projection

from ..mesh import Mesh, remove_th

def intg_th(projection: Projection) -> Projection:

    mesh_flat: Mesh = remove_th(projection.mesh)
    projection_intg_th: Projection = Projection(mesh_flat)
    
    proj_col_items: list = sorted(projection.cols.items())
    for proj_col_key, proj_col in proj_col_items:
        assert(proj_col.is_lf)
        
        [ndof_x, ndof_y] = proj_col.ndofs[:]
        col_intg_th: np.ndarray = projection.col_intg_th(proj_col_key)
        
        # There is only one cell per column
        cell_key: int = list(projection_intg_th.cols[proj_col_key].cells.keys())[0]
        projection_intg_th.cols[proj_col_key].cells[cell_key].vals = np.reshape(col_intg_th, [ndof_x, ndof_y, 1])
            
    return projection_intg_th