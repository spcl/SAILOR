#!/usr/bin/env bash

####################
## Configurations ##
####################

WITH_PLUTO=true
WITH_PLUTO_FUSED=false
WITH_POLLY=true
WITH_POLLY_FUSED=false
WITH_GT4PY_I=false
WITH_GT4PY_K=false
WITH_GT4PY_DACE=false
WITH_DACE_RELEASE=true
WITH_NODOACROSS=true
WITH_NODOACROSS_INCREMENT=false
WITH_DOACROSS_COLLAPSE=false
WITH_DOACROSS_DYNAMIC=false
WITH_DOACROSS_STATIC=false
WITH_DOACROSS_GUIDED=false
WITH_DOACROSS_INCREMENT_DYNAMIC=false
WITH_DOACROSS_INCREMENT_STATIC=true
WITH_DOACROSS_INCREMENT_GUIDED=false

# Number of measurements per benchmark.
T_REPS=1000

# Test configurations to run (number of threads, and problem sizes)
#N_THREADS=(256 128 64 32 16 8)
N_THREADS=(128)
#P_SIZES=(32,32,80 32,32,180 64,64,80 64,64,180 128,128,80 128,128,180 256,256,80 256,256,180)
P_SIZES=(32,32,180 64,64,180, 128,128,180)
#P_SIZES=(32,32,180 64,64,80 64,64,180 128,128,80 128,128,180 256,256,80 256,256,180)
#P_SIZES=(64,64,180 128,128,80 128,128,180 256,256,80 256,256,180)

##################
## System setup ##
##################

# Use the hostname to make sure different hosts sharing file systems do not
# clash while running benchmarks concurrently.
HNAME=$(hostname)
mkdir -p C_versions/build/$HNAME
mkdir -p C_versions/pluto/$HNAME

# Configure node
module load gcc/10.2.0
module load boost
module load cuda

# Configure DaCe
export DACE_library_blas_default_implementation="MKL"
export DACE_profiling_status="false"
export DACE_compiler_cpu_executable="clang++"
export DACE_cache="unique"

# Configure GT4Py
export GT_CACHE_DIR_NAME=".gt_cache_${HNAME}"

########################
## Perform benchmarks ##
########################

for ps in ${P_SIZES[@]}; do
    IFS=',' read I J K <<< "${ps}"
    echo -e "Running with problem size I=${I}, J=${J}, K=${K}"
    for nt in ${N_THREADS[@]}; do
        export OMP_NUM_THREADS=$nt
        echo -e "\tRunning with ${nt} threads"

        if $WITH_POLLY; then
            # Build polly
            clang -O3 -march=native -mllvm -polly -mllvm -polly-parallel -fopenmp=libomp -lm -fopenmp -lomp C_versions/vadv_polly_helped.c -o C_versions/build/$HNAME/vadv_polly_helped
            # Run
            echo "Polly"
            ./C_versions/build/$HNAME/vadv_polly_helped $I $J $K $T_REPS 2>/dev/null
            #polly_out=$(./C_versions/build/$HNAME/vadv_polly_helped $I $J $K $T_REPS 2>/dev/null)
            #echo -e "\t\tPolly: \t\t\t${polly_out}"
        fi

        if $WITH_POLLY_FUSED; then
            # Build polly fused
            clang -O3 -march=native -mllvm -polly -mllvm -polly-parallel -fopenmp=libomp -lm -fopenmp -lomp C_versions/vadv_polly_mega_helped.c -o C_versions/build/$HNAME/vadv_polly_mega_helped
            # Run
            polly_fused_out=$(./C_versions/build/$HNAME/vadv_polly_mega_helped $I $J $K $T_REPS 2>/dev/null)
            echo -e "\t\tPolly (Fused): \t\t${polly_fused_out}"
        fi

        if $WITH_PLUTO; then
            # Build pluto
            polycc C_versions/vadv_polly_helped.c --parallel --multipar -q -o C_versions/pluto/$HNAME/pluto.c &> /dev/null
            clang -O3 -march=native -fopenmp=libomp -lm -fopenmp -lomp C_versions/pluto/$HNAME/pluto.c -o C_versions/build/$HNAME/pluto
            rm -f pluto.pluto.cloog
            # Run
            echo "Pluto"
            ./C_versions/build/$HNAME/pluto $I $J $K $T_REPS 2>/dev/null
            #pluto_out=$(./C_versions/build/$HNAME/pluto $I $J $K $T_REPS 2>/dev/null)
            #echo -e "\t\tPluto: \t\t\t${pluto_out}"
        fi

        if $WITH_PLUTO_FUSED; then
            # Build pluto fused
            polycc C_versions/vadv_polly_mega_helped.c --parallel --multipar -q -o C_versions/pluto/$HNAME/pluto_fused.c &> /dev/null
            clang -O3 -march=native -fopenmp=libomp -lm -fopenmp -lomp C_versions/pluto/$HNAME/pluto_fused.c -o C_versions/build/$HNAME/pluto_fused
            rm -f pluto_fused.pluto.cloog
            # Run
            pluto_fused_out=$(./C_versions/build/$HNAME/pluto_fused $I $J $K $T_REPS 2>/dev/null)
            echo -e "\t\tPluto (Fused): \t\t${pluto_fused_out}"
        fi

        if $WITH_GT4PY_K; then
            gt_kfirst_out=$(conda run -n sc24_py310 python vadv_gt4py.py $I $J $K $T_REPS -b gt:cpu_kfirst 2>/dev/null)
            echo -e "\t\tGT4Py (kfirst): \t${gt_kfirst_out}"
        fi

        if $WITH_GT4PY_I; then
            gt_ifirst_out=$(conda run -n sc24_py310 python vadv_gt4py.py $I $J $K $T_REPS -b gt:cpu_ifirst 2>/dev/null)
            echo -e "\t\tGT4Py (ifirst): \t${gt_ifirst_out}"
        fi

        if $WITH_GT4PY_DACE; then
            gt_dace_out=$(conda run -n sc24_py310 python vadv_gt4py.py $I $J $K $T_REPS -b dace:cpu 2>/dev/null)
            echo -e "\t\tGT4Py (DaCe): \t${gt_dace_out}"
        fi

        if $WITH_DACE_RELEASE; then
            echo "Dace Release"
            conda run -n sc24_releases python vadv_dace.py $I $J $K $T_REPS 2>/dev/null
            #nodoacross_out=$(conda run -n sc24_releases python vadv_dace.py $I $J $K $T_REPS 2>/dev/null | grep 'auto_optimized' | awk '{print $2 " " $3}')
            #echo -e "\t\tDaCe Released: \t\t${nodoacross_out}"
        fi

        if $WITH_NODOACROSS; then
            echo "SAILOR"
            conda run -n sc24_py312 python do_opt.py $I $J $K $T_REPS 2>/dev/null
            #nodoacross_out=$(conda run -n sc24_py312 python do_opt.py $I $J $K $T_REPS 2>/dev/null | grep 'sc24_vadv_vadv_dace_vadv' | awk '{print $2 " " $3}')
            #echo -e "\t\tDaCe Autoopt: \t\t${nodoacross_out}"
        fi

        if $WITH_NODOACROSS_INCREMENT; then
            nodoacross_inc_out=$(conda run -n sc24_py312 python do_opt.py --increment $I $J $K $T_REPS 2>/dev/null | grep 'sc24_vadv_vadv_dace_vadv' | awk '{print $2 " " $3}')
            echo -e "\t\tNon-Doacross (Incr): \t${nodoacross_inc_out}"
        fi

        if $WITH_DOACROSS_DYNAMIC; then
            doacross_out=$(conda run -n sc24_py312 python do_opt.py --doacross --schedule=dynamic $I $J $K $T_REPS 2>/dev/null | grep 'sc24_vadv_vadv_dace_vadv' | awk '{print $2 " " $3}')
            echo -e "\t\tDoacross (Dynamic): \t${doacross_out}"
        fi

        if $WITH_DOACROSS_STATIC; then
            doacross_out=$(conda run -n sc24_py312 python do_opt.py --doacross --schedule=static $I $J $K $T_REPS 2>/dev/null | grep 'sc24_vadv_vadv_dace_vadv' | awk '{print $2 " " $3}')
            echo -e "\t\tDoacross (Static): \t${doacross_out}"
        fi

        if $WITH_DOACROSS_GUIDED; then
            doacross_out=$(conda run -n sc24_py312 python do_opt.py --doacross --schedule=guided $I $J $K $T_REPS 2>/dev/null | grep 'sc24_vadv_vadv_dace_vadv' | awk '{print $2 " " $3}')
            echo -e "\t\tDoacross (Guided): \t${doacross_out}"
        fi

        if $WITH_DOACROSS_COLLAPSE; then
            doacross_collapse_out=$(conda run -n sc24_py312 python do_opt.py --doacross --collapse $I $J $K $T_REPS 2>/dev/null | grep 'sc24_vadv_vadv_dace_vadv' | awk '{print $2 " " $3}')
            echo -e "\t\tDoacross (Collapsing): \t${doacross_collapse_out}"
        fi

        if $WITH_DOACROSS_INCREMENT_DYNAMIC; then
            doacross_increment_out=$(conda run -n sc24_py312 python do_opt.py --doacross --increment --schedule=dynamic $I $J $K $T_REPS 2>/dev/null | grep 'sc24_vadv_vadv_dace_vadv' | awk '{print $2 " " $3}')
            echo -e "\t\tDoacross (Incr Dy.): \t${doacross_increment_out}"
        fi

        if $WITH_DOACROSS_INCREMENT_STATIC; then
            echo "SAILOR (Pipelined)"
            conda run -n sc24_py312 python do_opt.py --doacross --collapse --schedule=static $I $J $K $T_REPS 2>/dev/null
            #doacross_collapse_out=$(conda run -n sc24_py312 python do_opt.py --doacross --collapse --schedule=static $I $J $K $T_REPS 2>/dev/null | grep 'sc24_vadv_vadv_dace_vadv' | awk '{print $2 " " $3}')
            #echo -e "\t\tDoacross (Incr St.): \t${doacross_collapse_out}"
        fi

        if $WITH_DOACROSS_INCREMENT_GUIDED; then
            doacross_increment_out=$(conda run -n sc24_py312 python do_opt.py --doacross --increment --schedule=guided $I $J $K $T_REPS 2>/dev/null | grep 'sc24_vadv_vadv_dace_vadv' | awk '{print $2 " " $3}')
            echo -e "\t\tDoacross (Incr Gd.): \t${doacross_increment_out}"
        fi

        echo -e "------------------------------------------------"
    done
    echo -e "================================================"
done

echo "done"
