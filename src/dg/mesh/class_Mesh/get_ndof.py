# Standard Library Imports

# Third-Party Library Imports

# Local Library Imports
from class_Column import Column

# Relative Imports

def get_ndof(self) -> int:
    mesh_ndof: int = 0
    for _, col in self.cols.items():
        if col.is_lf:
            [ndof_x, ndof_y] = col.ndofs[:]
            
            for _, cell in col.cells.items():
                if cell.is_lf:
                    [ndof_th] = cell.ndofs[:]
                    
                    mesh_ndof += int(ndof_x * ndof_y * ndof_th)
    
    return int(mesh_ndof)