#!/bin/bash

## Stop at the first failed step
set -euo pipefail

EXPERIMENT_NAME="Manufactured Solution"

N_PROCS=1
while getopts n: flag
do
    case "${flag}" in
        n) N_PROCS=${OPTARG};;
        *) printf "Usage: %s [-n N_PROCS]\n" "$0"; exit 1;;
    esac
done

OUT_DIR="out"
mkdir -p ${OUT_DIR}

printf "~~~ Executing ${EXPERIMENT_NAME} experiment with ${N_PROCS} processes... ~~~\n\n"

printf " ~~ Initiating problem visualization generation... ~~ \n\n"
mpirun -n 1 python generate_problem_visualizations.py --o "${OUT_DIR}"
printf " ~~ Completed problem visualization generation! ~~ \n\n"

printf " ~~ Initiating radiative transfer numerical solve... ~~ \n\n"
mpirun --use-hwthread-cpus -n "${N_PROCS}" python experiment.py --o "${OUT_DIR}"
printf " ~~ Completed radiative transfer numerical solve! ~~ \n\n"

printf " ~~ Initiating error histories generation... ~~ \n\n"
mpirun --use-hwthread-cpus -n "${N_PROCS}" python generate_error_history.py --o "${OUT_DIR}"
printf " ~~ Error histories generated! ~~ \n\n"

printf " ~~ Initiating linear solve visualizations... ~~ \n\n"
mpirun --use-hwthread-cpus -n 1 python generate_linear_solve_visualizations.py --o "${OUT_DIR}"
printf " ~~ Completed linear solve visualization! ~~ \n\n"

printf " ~~ Initiating experiment visualization generation... ~~ \n\n"
mpirun --use-hwthread-cpus -n 1 python generate_experiment_visualizations.py --o "${OUT_DIR}"
printf " ~~ Completed experiment visualization generation! ~~ \n\n"

printf " ~~ Initiating convergence plot generation... ~~ \n\n"
mpirun --use-hwthread-cpus -n 1 python generate_convergence_plot.py --o "${OUT_DIR}"
printf " ~~ Completed convergence plot generation! ~~ \n\n"

printf "~~~ ${EXPERIMENT_NAME} experiment complete! ~~~\n\n"
