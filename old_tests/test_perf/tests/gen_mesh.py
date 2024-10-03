import sys

sys.path.append("../../src")
import dg.mesh as ji_mesh

def gen_mesh(Ls, pbcs, ndofs, has_th):
    """
    Generates a mesh to be used as the base mesh for a test.
    """
    
    # Create the base mesh which will be refined in each trial.
    mesh = ji_mesh.Mesh(Ls     = Ls,
                        pbcs   = pbcs,
                        ndofs  = ndofs,
                        has_th = has_th)
    
    # Refine the mesh for initial trial
    # Uniform: Four columns, four cells per column
    for _ in range(0, 2):
        mesh.ref_mesh(kind = "ang")
    for _ in range(0, 1):
        mesh.ref_mesh(kind = "spt")

    return mesh
