#!/bin/bash

while getopts n: flag
do
    case "${flag}" in
        n) n_procs=${OPTARG};;
    esac
done

OUT_DIR="out"

printf "~~~ Executing Analytic Solution experiment with ${n_procs} processes... ~~~\n\n"

mpirun -n ${n_procs} python experiment.py --o ${OUT_DIR}

printf "~~~ Analytic Solution experiment complete! ~~~\n\n"
