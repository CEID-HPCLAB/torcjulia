#!/usr/bin/env bash

# This script reproduces all results for the first workload (Parallel Non-Linear Optimization using the Hooke-Jeeves Algorithm). 
# It runs the experiment with different worker configurations (i.e., number of MPI ranks), 
# while keeping the number of Julia worker threads per rank fixed at 2.
# For each configuration, it performs 5 benchmark runs and records the median wall-clock time in 
# a corresponding file (experiments/results/torcjulia/hooke-jeeves.txt).

set -euo pipefail
export LC_ALL=C

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../../../experiments" && pwd)"

JULIA_BIN="${JULIA_BIN:-$(command -v julia || true)}"

if [[ -z "$JULIA_BIN" ]]; then
    echo "ERROR: Julia executable not found in PATH."
    echo "Install Julia and ensure it is available in your PATH, or set the JULIA_BIN environment variable to the path of the Julia executable."
    exit 1
fi

MPIEXECJL="${MPIEXECJL:-$(command -v mpiexecjl || true)}"

if [[ -z "$MPIEXECJL" || ! -x "$MPIEXECJL" ]]; then
    echo "ERROR: mpiexecjl not found or not executable."
    echo "Add its directory to PATH or set MPIEXECJL to its full path."
    exit 1
fi

NUM_THREADS=3
NUM_MPI_RANKS=(1 2 4 8 12 16 24 32 48 64)

BENCHMARK_RUNS=5

JULIA_VERSION=$("$JULIA_BIN" --version | awk '{print $3}')

OUTPUT_FILE="$PROJECT_DIR/results/torcjulia/hooke-jeeves.txt"

mkdir -p "$(dirname "$OUTPUT_FILE")"

: > "$OUTPUT_FILE"

printf 'Running Workload 1: %s.\n' \
       "Parallel Non-Linear Optimization using the Hooke-Jeeves Algorithm"

for NUM_RANKS in "${NUM_MPI_RANKS[@]}"; do
    
    WALL_CLOCK_TIMES=()

    if [[ "$NUM_RANKS" -eq 1 ]]; then
        THREADS_PER_RANK=$((NUM_THREADS - 1))
        WORKER_THREADS="$THREADS_PER_RANK"
    else
        THREADS_PER_RANK="$NUM_THREADS"
        WORKER_THREADS=$((THREADS_PER_RANK - 1))
    fi

    if [[ "$(printf '%s\n' "1.12.0" "$JULIA_VERSION" | sort -V | head -n1)" == "1.12.0" ]]; then
        THREADS_ARG="$THREADS_PER_RANK,0"
    else
        THREADS_ARG="$THREADS_PER_RANK"
    fi

    if [[ "$NUM_RANKS" -eq 32 || "$NUM_RANKS" -eq 48 || "$NUM_RANKS" -eq 64 ]]; then
        WORK_STEALING="false"
    else
        WORK_STEALING="true"
    fi  

    echo

    for ((i = 1; i <= BENCHMARK_RUNS; i++)); do
        
        if [[ "$NUM_RANKS" -eq 1 ]]; then
            printf "Benchmark Run %d/%d: %d MPI process with %d Julia threads (%d worker thread(s)).\n" \
                "$i" "$BENCHMARK_RUNS" "$NUM_RANKS" "$THREADS_PER_RANK" \
                "$WORKER_THREADS"
        else
            printf "Benchmark Run %d/%d: %d MPI processes with %d Julia threads per process (%d worker thread(s), 1 manager thread).\n" \
                "$i" "$BENCHMARK_RUNS" "$NUM_RANKS" "$THREADS_PER_RANK" \
                "$WORKER_THREADS"
        fi

        TIME_OUTPUT=$(
             WORK_STEALING="$WORK_STEALING" \
            "$MPIEXECJL" -n "$NUM_RANKS" \
                "$JULIA_BIN" -O3 \
                --project="$PROJECT_DIR" \
                --threads="$THREADS_ARG" \
                "$PROJECT_DIR/hooke-jeeves/torcjulia/main.jl"
        )
        
        WALL_CLOCK_TIME=$(printf '%s\n' "$TIME_OUTPUT" | sed -n 's/.*Time taken[^:]*: *\([0-9.]*\) seconds.*/\1/p' | tail -n 1)
        
        WALL_CLOCK_TIMES+=("$WALL_CLOCK_TIME")

    done
    
    MEDIAN_TIME=$(printf '%s\n' "${WALL_CLOCK_TIMES[@]}" | sort -n |
                  awk '{a[NR]=$1} END {printf "%.4f\n", (a[int((NR+1)/2)]+a[int((NR+2)/2)])/2}')
    
    printf 'MPI Processes: %s, Worker Threads: %s | Wall-Clock Time: %s seconds\n' \
        "$NUM_RANKS" "$WORKER_THREADS" "$MEDIAN_TIME" >> "$OUTPUT_FILE"
    
    echo

done

echo "All benchmark runs completed. Results saved to $OUTPUT_FILE."