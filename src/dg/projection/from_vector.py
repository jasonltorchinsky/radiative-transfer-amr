# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports

# Relative Imports
from ..mesh import Mesh
from ..matrix import get_idx_inv_map

def from_vector(self, mesh: Mesh, vec: np.ndarray) -> None:
    col_items: list = sorted(mesh.cols.items())

    g_idx: int = 0 # Starting index of the current cell matrix
    for col_key, col in col_items:
        assert(col.is_lf)
        
        [ndof_x, ndof_y] = col.ndofs[:]
        cell_items: list = sorted(col.cells.items())
        
        for cell_key, cell in cell_items:
            assert(cell.is_lf)

            [ndof_th] = cell.ndofs
            
            cell_vals: np.ndarray = np.zeros([ndof_x, ndof_y, ndof_th])
            
            ija = get_idx_inv_map(ndof_x, ndof_y, ndof_th)
            
            cell_ndof: int = ndof_x * ndof_y * ndof_th
            cell_vec: np.ndarray = vec[g_idx:g_idx + cell_ndof]
            for beta in range(0, cell_ndof):
                [ii, jj, aa] = ija(beta)
                cell_vals[ii, jj, aa] = cell_vec[beta]
                
            self.cols[col_key].cells[cell_key].vals = cell_vals[:,:,:]
            
            g_idx += cell_ndof

    return self