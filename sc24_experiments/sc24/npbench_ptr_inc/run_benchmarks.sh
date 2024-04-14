#!/usr/bin/env bash

WITH_SEQUENTIAL=true
WITH_PARALLEL=false

T_REPS=1000

#PROBLEM_SIZES=('S' 'M' 'L' 'paper')
PROBLEM_SIZES=('M')

# Configure node
module load gcc/10.2.0
#module load intel-oneapi-compilers

# Configure DaCe
export DACE_library_blas_default_implementation="MKL"
export DACE_profiling_status="false"
export DACE_compiler_cpu_executable="g++"
export DACE_cache="unique"
export DACE_optimizer_visualizie_sdfv="false"
export KMP_DUPLICATE_LIB_OK=TRUE

for ps in ${PROBLEM_SIZES[@]}; do
    echo -e "Running problem size ${ps}"

    for f in $(find benchmarks -name '*.py'); do
        if $WITH_SEQUENTIAL; then
            echo -e "Running ${f} with no incrementation, sequentially"
            conda run -n sc24_py312 python $f -p $ps -r $T_REPS 2>/dev/null
            echo -e "Running ${f} WITH incrementation, sequentially"
            conda run -n sc24_py312 python $f -p $ps --increment -r $T_REPS 2>/dev/null
        fi

        if $WITH_PARALLEL; then
            echo -e "Running ${f} with no incrementation, in parallel"
            conda run -n sc24_py312 python $f -p $ps --parallel -r $T_REPS 2>/dev/null
            echo -e "Running ${f} WITH incrementation, in parallel"
            conda run -n sc24_py312 python $f -p $ps --parallel --increment -r $T_REPS 2>/dev/null
        fi
    done
done
