`rt`
================================================================================

This subdirectory contains the source code regarding numerically solving the steady-state monochromatic radiative transfer equation.

Source Files
--------------------------------------------------------------------------------

The source files contained in the `rt` subdirectory are:

- `boundary_conditions_vector` : Calculates the vecotr that holds the inflow boundary conditions.
- `boundary_convection_matrix` : Calculates the boundary convection matrix.
- `class_Problem` : The `Problem` class stores information about the problem, e.g., the extinction coefficient, scattering coefficient, and scattering phase function.
- `forcing_vector` : Calculates the forcing vector.
- `get_Eth` : Calculates the $\pmb{\mathcal{E}}^{\theta}$ matrices used in the boundary convection matrix.
- `get_Ex` : Calculates the $\pmb{\mathcal{E}}^{x}$ matrices used in the boundary convection matrix.
- `get_Ey` : Calculates the $\pmb{\mathcal{E}}^{y}$ matrices used in the boundary convection matrix.
- `interior_convection_matrix` : Calculates the interior convection matrix.
- `mass_matrix` : Calculates the extinction (mass) matrix.
- `preconditioner_matrix` : Calculates the $extinction - scattering$ matrix, which could be used as the preconditioner.
- `scattering_matrix` : Calculates the scattering matrix.
- `solve` : Numerically solves the steady-state monochromatic radiative transfer equation.