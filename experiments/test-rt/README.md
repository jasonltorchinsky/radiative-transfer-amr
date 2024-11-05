`test-rt`
================================================================================

This subdirectory contains the code and scripts used to test source code in the `src/rt` subdirectory. We have included it in the `experiments` subdirectory as it requires MPI.

The user may think of this test as a simplified version of the numerical experiments for in `manufactured-solution`, `one-weak-scatterer`, and `two-strong-scatterers` subdirectories. As such, we refer to this test in the code as an 'experiment'.

Running the Test
--------------------------------------------------------------------------------

To run the 'test-rt' test, perform the following steps:

1. Ensure that you have created and have activated the included `conda` environment by running the following commands in the root directory:

```
conda env create -f environment.yml
conda activate radiative-transfer-amr
```

2. Ensure that the run script `run.sh` is executable:

```
chmod +x run.sh
```

3. Execute the run script using a specified number ($N$) of MPI tasks:

```
./run.sh -n N
```

Modifying the Test
--------------------------------------------------------------------------------

There are three primary files utilized in initializing and executing the test that may be modified:
- `input.json` : Contains parameters for initializing the test.
    - `mesh_params` : Parameters for the initial mesh.
        - `Ls` : Length of the spatial domain in the $x$- and $y$- directions.
        - `pbcs` : Directions in which there are periodic boundary conditions (unverified).
        - `has_th` : Specifies that the mesh has an angular dimension.
        - `ndofs` : Specifies the number of degrees of freedom in the $x$-, $y$-, and $\theta$-directions.
        - `nref_ang` : Specifies the number of times the *h*-refine the initial mesh in the angular dimension.
        - `nref_spt` : Specifies the number of times the *h*-refine the initial mesh in the spatial dimensions.
- `problem.py` : Defines functions used the the extinction coefficient, scattering coefficent, etc.
- `run.sh` : Run script to execute the experiment.
    - `EXPERIMENT_NAME` : Only used in `printf` statements of the run script.
    - `OUT_DIR` : (Relative) Path for outputting files from the experiment.