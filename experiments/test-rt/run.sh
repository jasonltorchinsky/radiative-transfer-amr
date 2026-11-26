#!/bin/bash

## Stop at the first failed step
set -euo pipefail

EXPERIMENT_NAME="test_rt"

n_procs=1
while getopts n: flag
do
    case "${flag}" in
        n) n_procs=${OPTARG};;
        *) printf "Usage: %s [-n N_PROCS]\n" "$0"; exit 1;;
    esac
done

OUT_DIR="out"

printf "~~~ Executing ${EXPERIMENT_NAME} experiment with ${n_procs} processes... ~~~\n\n"

printf " ~~ Initiating radiative transfer numerical solve... ~~ \n\n"
mpirun -n "${n_procs}" python experiment.py --o "${OUT_DIR}"
printf " ~~ Completed radiative transfer numerical solve! ~~ \n\n"

printf "~~~ ${EXPERIMENT_NAME} experiment complete! ~~~\n\n"
