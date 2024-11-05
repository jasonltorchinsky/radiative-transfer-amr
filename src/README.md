`src`
================================================================================

This subdirectory contains the source code implementing the algorithm.

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the `src` subdirectory is

```
src
|
|--- amr : Contains source code regarding adaptive mesh refinement, such as the Error_Indicator class, different ways to calculate error (e.g., analytic, high-resolution), and refining a mesh based on an error indicator.
|--- consts : Contains definitions for constants used throughout the code, such as data types (e.g., REAL) and numerical values (e.g., INF, EPS).
|--- dg : Contains source code regarding discontinuous Galerkin methods, such as the Mesh and Projection classes, methods for numerical quadrature, and methods for handling matrices across multiple MPI processes.
|--- rt : Contains source code regarding the numerical algorithm for solving the steady-state, monochromatic radiative transfer equation, including constructing the extinction (mass), scattering, and convection matrices, the Problem class, and solving the matrix system that arises.
|--- tools : Contains source code for generating relevant plots, e.g., of the *hp*-mesh, numerical solutions on the *hp*-mesh, and error indicators.
|--- utils : Contains source code for utility functions in the source code, such as printing messages to the command terminal from the ROOT MPI process.
```