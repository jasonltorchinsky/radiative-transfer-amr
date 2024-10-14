# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports

# Relative Imports
from .class_Projection import Projection

from ..mesh import Mesh
from ..mesh import from_file as mesh_from_file

def from_file(mesh_file_path: str = "mesh.json", 
              projection_file_path: str = "projection.npy") -> Projection:
    
    mesh: Mesh = mesh_from_file(mesh_file_path)
    proj_vec: np.ndarray = np.fromfile(projection_file_path)
    proj: Projection = Projection(mesh, None)
    proj.from_vector(proj_vec)

    return proj