# Standard Library Imports

# Third-Party Library Imports
import numpy as np

# Local Library Imports

# Relative Imports

def to_file(self, file_path: str = "projection.npy", **kwargs) -> None:
    default_kwargs: dict = {"write_mesh" : True,
                            "mesh_file_path" : "mesh.json"}
    kwargs: dict = {**default_kwargs, **kwargs}

    if kwargs["write_mesh"]:
        self.mesh.to_file(kwargs["mesh_file_path"])

    proj_vec: np.ndarray = self.to_vector()
    proj_vec.tofile(file_path)