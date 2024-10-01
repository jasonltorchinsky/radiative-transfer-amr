#!/bin/bash

EXPERIMENT_NAME="Analytic Solution"

while getopts n: flag
do
    case "${flag}" in
        n) n_procs=${OPTARG};;
    esac
done

OUT_DIR="out"


printf "~~~ Executing ${EXPERIMENT_NAME} experiment with ${n_procs} processes... ~~~\n\n"

printf " ~~ Initiating extinction, scattering coefficients, and scattering phase function plot creation ... ~~ \n\n"
eval 'python plot_kappa_sigma_phi.py --o "${OUT_DIR}"'
printf " ~~ Completed extinction, scattering coefficients, and scattering phase function plot creation! ~~ \n\n"

printf " ~~ Initiating analytic plot creation ... ~~ \n\n"
eval 'mpirun -n 1 python plot_u.py --o "${OUT_DIR}"'
printf " ~~ Completed analytic plot creation! ~~ \n\n"

#printf " ~~ Initiating radiative transfer numerical solve... ~~ \n\n"
#eval 'mpirun -n ${n_procs} python experiment.py --o "${OUT_DIR}"'
#printf " ~~ Completed radiative transfer numerical solve! ~~ \n\n"

printf "~~~ ${EXPERIMENT_NAME} experiment complete! ~~~\n\n"
