`dg`
================================================================================

This subdirectory contains the source code regarding the discontinuous Galerkin method.

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `dg` subdirectory is

```
dg
|
|--- matrix : Contains source code to handle matrices spread across multiple MPI processes.
|--- mesh : Contains source code to handle the *hp*-mesh.
|--- projection : Contains source code to handle functions projected to the *hp*-mesh.
|--- quadrature : Contains source code to handle numerical quadrature.
```