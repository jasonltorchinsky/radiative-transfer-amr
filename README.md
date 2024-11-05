Angular-Spatial *hp*-Adaptivity for Radiative Transfer with Discontinuous Galerkin Spectral Element Methods
================================================================================

This code was used to generate the results reported in

J. L. Torchinsky, S. Du, and S. N. Stechmann, Angular-Spatial *hp*-Adaptivity for Radiative Transfer with Discontinuous Galerkin Spectral Element Methods, [In Review], (2024).

Installation
--------------------------------------------------------------------------------

Upon cloning this repository, you may obtain the `conda` environment from the provided `environment.yml` file via executing the following command in the root directory:

```conda env create -f environment.yml```

There are several Bash scripts (files with extension `.sh`) included in this repository - especially in the `experiments` subdirectory - that are used to execute pre-written piece of code. To make such a script executable on your local machine, you use the `chmod` command, e.g.:

```chmod +x script_name.sh```

Subdirectory Structure
--------------------------------------------------------------------------------

The subdirectory structure for the root directory is

```
root
|
|--- experiments : Numerical experiments (as reported in the associated publication) and tests that require MPI.
|--- src : Source code for the implementation of the angular-spatial hp-adaptive algorithm for (steady-state, monochromatic) radiative transfer in two spatial dimensions and one angular dimension.
|--- tests : Unit tests that utilize the pytest testing framework.
```