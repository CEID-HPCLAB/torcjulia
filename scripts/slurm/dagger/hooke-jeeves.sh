#!/bin/bash -l

#SBATCH --job-name=hooke-jeeves
#SBATCH --nodes=<NODES>
#SBATCH --ntasks=<NTASKS>
#SBATCH --ntasks-per-node=<NTASKS_PER_NODE>
#SBATCH --ntasks-per-socket=<NTASKS_PER_SOCKET>
#SBATCH --cpus-per-task=<CPUS_PER_TASK>
#SBATCH --mem=<MEMORY>
#SBATCH --time=<TIME_LIMIT>
#SBATCH --partition=<PARTITION>
#SBATCH -A <ACCOUNT>

#SBATCH --output=./hooke-jeeves_%j.out
#SBATCH --error=./hooke-jeeves_%j.err

set -euo pipefail

module purge

# NOTE: Adapt the module names and versions below to match
# the software modules available on the target HPC cluster.
module load julia/1.12.4

export PATH="$HOME/.julia/bin:$PATH"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../../../experiments" && pwd)"

MAIN_SCRIPT="$SCRIPT_DIR/../../../experiments/hooke-jeeves/dagger/main.jl"

THREADS_ARG="${SLURM_CPUS_PER_TASK},0" # Assuming Julia version >= 1.12.0, otherwise use "${SLURM_CPUS_PER_TASK}"

NUM_RANKS="$SLURM_NTASKS"
NUM_THREADS="$SLURM_CPUS_PER_TASK"

export DAGGER_COOKIE=$(head -c 8 /dev/urandom | od -An -tx1 | tr -d ' \n')

export SLURM_JOB_SCHEDULING=true

if [[ "$NUM_RANKS" -eq 1 ]]; then
    USE_MAIN_THREAD=true
else
    USE_MAIN_THREAD=false
fi

export USE_MAIN_THREAD

TIME_OUTPUT=$(srun --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    --cpu-bind=cores --distribution=block:cyclic \
    julia -O3 \
    --project="$PROJECT_DIR" \
    --threads="$THREADS_ARG" \
    "$MAIN_SCRIPT" \
)

WALL_CLOCK_TIME=$(printf '%s\n' "$TIME_OUTPUT" | sed -n 's/.*Time taken[^:]*: *\([0-9.]*\) seconds.*/\1/p' | tail -n 1)

printf 'Julia Processes: %s, Julia Threads: %s | Wall-Clock Time: %s seconds\n' \
        "$NUM_RANKS" "$NUM_THREADS" "$WALL_CLOCK_TIME"