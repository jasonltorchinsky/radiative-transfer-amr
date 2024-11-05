#!/bin/bash

EXPERIMENT_NAME="Manufactured Solution"

while getopts n: flag
do
    case "${flag}" in
        n) N_PROCS=${OPTARG};;
    esac
done

OUT_DIR="out"
mkdir -p ${OUT_DIR}

printf "~~~ Executing ${EXPERIMENT_NAME} experiment with ${N_PROCS} processes... ~~~\n\n"

printf " ~~ Initiating problem visualization generation... ~~ \n\n"
eval 'mpirun -n 1 python generate_problem_visualizations.py --o '"${OUT_DIR}"
printf " ~~ Completed problem visualization generation! ~~ \n\n"

printf " ~~ Initiating radiative transfer numerical solve... ~~ \n\n"
eval 'mpirun --use-hwthread-cpus -n '"${N_PROCS}"' python experiment.py --o '"${OUT_DIR}"
printf " ~~ Completed radiative transfer numerical solve! ~~ \n\n"

printf " ~~ Initiating error histories generation... ~~ \n\n"
eval 'mpirun --use-hwthread-cpus -n '"${N_PROCS}"' python generate_error_history.py --o '"${OUT_DIR}"
printf " ~~ Error histories generated! ~~ \n\n"

printf " ~~ Initiating linear solve visualizations... ~~ \n\n"
eval 'mpirun --use-hwthread-cpus -n 1 python generate_linear_solve_visualizations.py --o '"${OUT_DIR}"
printf " ~~ Completed linear solve visualization! ~~ \n\n"

printf " ~~ Initiating experiment visualization generation... ~~ \n\n"
eval 'mpirun --use-hwthread-cpus -n 1 python generate_experiment_visualizations.py --o '"${OUT_DIR}"
printf " ~~ Completed experiment visualization generation! ~~ \n\n"

printf " ~~ Initiating convergence plot generation... ~~ \n\n"
eval 'mpirun --use-hwthread-cpus -n 1 python generate_convergence_plot.py --o '"${OUT_DIR}"
printf " ~~ Completed convergence plot generation! ~~ \n\n"

printf "~~~ ${EXPERIMENT_NAME} experiment complete! ~~~\n\n"
