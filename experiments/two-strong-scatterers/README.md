`two-strong-scatterers`
================================================================================

This subdirectory contains the code and scripts used to generate the 'Two Strong Scatterers' numerical experiment results reported in

J. L. Torchinsky, S. Du, and S. N. Stechmann, Angular-Spatial *hp*-Adaptivity for Radiative Transfer with Discontinuous Galerkin Spectral Element Methods, [In Review], (2024).

Note: There are difficulties with the stability of the numerical solve for the `hp-amr-ang-p-uni-spt` refinement strategy. This by be alleviated by changing the settings for the iterative solver, whether that be:
- the Krylov method or preconditioner used (as specified in `input.json`), or;
- the `max_it` or `GMRESRestart` parameters specified in `src/rt/solve.py`.

Running the Numerical Experiment
--------------------------------------------------------------------------------

To run the 'Two Strong Scatterers' numerical experiment, perform the following steps:

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

Modifying the Numerical Experiment
--------------------------------------------------------------------------------

There are three primary files utilized in initializing and executing the numerical experiment that may be modified:
- `input.json` : Contains parameters for initializing and executing the experiment.
    - `seed` : The seed used for RNG throughout the experiment (in pratice, only for the angular-spatial steering criterion). It is changed for each refinement in a known way. Entering `0` uses a pseudo-random seed, which is recorded on output for reproducibility.
    - `ndof_output_ratio` : Frequency for calculating error, generating experiment visualizations, etc., based on the ratio of the number of degrees of freedom of the previous mesh for which these were performed.
    - `solver_params` : Parameters for the `PETSc` linear solver used to solve the radiative transfer equation.
        - `ksp_type` : The Krylov method name for `PETSc` KSP object.
        - `pc_type` : The preconditioner name for the `PETSc` PC object.
    - `stopping_conditions` : Conditions for which each refinement strategy investigated in the experiment will conclude.
        - `max_ndof` : Stop if attempting a trial with a mesh with at least this many degrees of freedom.
        - `max_ntrial` : Stop if attempting a trial beyond this many.
        - `min_err` : Stop if the numerical solution has at most this much error.
    - `mesh_params` : Parameters for the initial mesh.
        - `Ls` : Length of the spatial domain in the $x$- and $y$- directions.
        - `pbcs` : Directions in which there are periodic boundary conditions (unverified).
        - `has_th` : Specifies that the mesh has an angular dimension.
        - `ndofs` : Specifies the number of degrees of freedom in the $x$-, $y$-, and $\theta$-directions.
        - `nref_ang` : Specifies the number of times the *h*-refine the initial mesh in the angular dimension.
        - `nref_spt` : Specifies the number of times the *h*-refine the initial mesh in the spatial dimensions.
    - `hr_err_params` : Parameters used in calculating the high-resolution error.
        - `ang_ref_offset` : Number of times to *p*-refine the mesh in angle to obtain the mesh on which the high-resolution solution is calculated.
        - `spt_ref_offset` : Number of times to *p*-refine the mesh in spatial to obtain the mesh on which the high-resolution solution is calculated.
    - `hp-amr-spt-p-uni-ang`: Parameters for the adaptive spatial *hp*-refinement, uniform angular *p*-refinement strategy.     
        - `ang` : Parameters for the angular part of the adaptive spatial *hp*-refinement, uniform angular *p*-refinement strategy.
            - `ref_kind` : Refinement kind (`ang`, `spt`, or `all`).
            - `ref_form` : Refinement form (`h`, `p`, or `hp`).
            - `ref_tol` : Refinement tolerances in [angle, space], each between 0. and 1. (inclusive).
        - `spt` : Parameters for the spatial part of the adaptive spatial *hp*-refinement, uniform angular *p*-refinement strategy.
            - `ref_kind` : Refinement kind (`ang`, `spt`, or `all`).
            - `ref_form` : Refinement form (`h`, `p`, or `hp`).
            - `ref_tol` : Refinement tolerances in [angle, space], each between 0. and 1. (inclusive).
    - `hp-amr-ang-p-uni-spt`: Parameters for the adaptive angular *hp*-refinement, uniform spatial *p*-refinement strategy.     
        - `ang` : Parameters for the angular part of the adaptive angular *hp*-refinement, uniform spatial *p*-refinement strategy.
            - `ref_kind` : Refinement kind (`ang`, `spt`, or `all`).
            - `ref_form` : Refinement form (`h`, `p`, or `hp`).
            - `ref_tol` : Refinement tolerances in [angle, space], each between 0. and 1. (inclusive).
        - `spt` : Parameters for the spatial part of the adaptive angular *hp*-refinement, uniform spatial *p*-refinement strategy.
            - `ref_kind` : Refinement kind (`ang`, `spt`, or `all`).
            - `ref_form` : Refinement form (`h`, `p`, or `hp`).
            - `ref_tol` : Refinement tolerances in [angle, space], each between 0. and 1. (inclusive).
    - `hp-amr-ang-hp-amr-spt`: Parameters for the adaptive angular *hp*-refinement, adaptive spatial *hp*-refinement strategy.  
        - `ang` : Parameters for the angular part of the adaptive angular *hp*-refinement, adaptive spatial *hp*-refinement strategy.
            - `ref_kind` : Refinement kind (`ang`, `spt`, or `all`).
            - `ref_form` : Refinement form (`h`, `p`, or `hp`).
            - `ref_tol` : Refinement tolerances in [angle, space], each between 0. and 1. (inclusive).
        - `spt` : Parameters for the spatial part of the adaptive angular *hp*-refinement, adaptive spatial *hp*-refinement strategy.
            - `ref_kind` : Refinement kind (`ang`, `spt`, or `all`).
            - `ref_form` : Refinement form (`h`, `p`, or `hp`).
            - `ref_tol` : Refinement tolerances in [angle, space], each between 0. and 1. (inclusive).
- `problem.py` : Defines functions used the the extinction coefficient, scattering coefficent, etc.
- `run.sh` : Run script to execute the experiment.
    - `EXPERIMENT_NAME` : Only used in `printf` statements of the run script.
    - `OUT_DIR` : (Relative) Path for outputting files from the experiment.