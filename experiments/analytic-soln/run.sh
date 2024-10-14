#!/bin/bash

EXPERIMENT_NAME="analytic-solution"

while getopts n: flag
do
    case "${flag}" in
        n) N_PROCS=${OPTARG};;
    esac
done

OUT_DIR="error-high-resolution"

printf "~~~ Executing ${EXPERIMENT_NAME} experiment with ${N_PROCS} processes... ~~~\n\n"

#printf " ~~ Initiating extinction, scattering coefficients, and scattering phase function plot creation ... ~~ \n\n"
#eval 'python plot_kappa_sigma_phi.py --o "${OUT_DIR}"'
#printf " ~~ Completed extinction, scattering coefficients, and scattering phase function plot creation! ~~ \n\n"

#printf " ~~ Initiating analytic plot creation ... ~~ \n\n"
#eval 'mpirun -n 1 python plot_u.py --o "${OUT_DIR}"'
#printf " ~~ Completed analytic plot creation! ~~ \n\n"

printf " ~~ Initiating radiative transfer numerical solve... ~~ \n\n"
eval 'mpirun --use-hwthread-cpus -n '"${N_PROCS}"' python experiment.py --o '"${OUT_DIR}"
printf " ~~ Completed radiative transfer numerical solve! ~~ \n\n"

for STRAT_DIR in ${OUT_DIR}/*/ ;
do
    for TRIAL_DIR in ${STRAT_DIR}*/ ;
    do
        printf "  ~ Initiating error indicator obtainment... ~  \n\n"
        eval 'mpirun --use-hwthread-cpus -n '"${N_PROCS}"' python get_err_ind_trial.py --o '"${TRIAL_DIR}"''
        printf "  ~ Error indicators obtained! ~  \n\n"

        printf " ~~ Initiating trial visualizations... ~~ \n\n"
        eval 'mpirun --use-hwthread-cpus -n 1 python visualize_trial.py --o '"${TRIAL_DIR}"
        printf " ~~ Completed trial visualizations... ~~ \n\n"
    done
done

printf " ~~ Initiating plotting convergence rates... ~~ \n\n"
eval 'python plot_convergence.py --o '"${OUT_DIR}"
printf " ~~ Completed plotting convergence rates! ~~ \n\n"

printf "~~~ ${EXPERIMENT_NAME} experiment complete! ~~~\n\n"
