# Standard Library Imports
import json

# Third-Party Library Imports

# Local Library Imports
from dg.mesh import Mesh

# Relative Imports

## Read input files - hardcoded file names
input_file = open("input.json")
input_dict: dict = json.load(input_file)
input_file.close()

mesh_params: dict = input_dict["mesh_params"]

## Setup the Mesh
mesh: Mesh = Mesh(Ls = mesh_params["Ls"],
                  pbcs = mesh_params["pbcs"],
                  has_th = mesh_params["has_th"],
                  ndofs = mesh_params["ndofs"])

for _ in range(0, mesh_params["nref_ang"]):
    mesh.ref_mesh(kind = "ang", form = "h")
    
for _ in range(0, mesh_params["nref_spt"]):
    mesh.ref_mesh(kind = "spt", form = "h")