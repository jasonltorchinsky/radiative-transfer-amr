#!/bin/bash

while getopts n:t: flag
do
    case "${flag}" in
        n) n_procs=${OPTARG};;
        t) test_num=${OPTARG};;
    esac
done

echo "Running test ${test_num} with ${n_procs} processes..."

cp "test_${test_num}.py" "test_temp.py"
mpirun -n ${n_procs} python gen_test_figs.py --test_num ${test_num}
