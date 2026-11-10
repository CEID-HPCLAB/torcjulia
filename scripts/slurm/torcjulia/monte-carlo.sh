#!/bin/bash -l

#SBATCH --job-name=monte-carlo
#SBATCH --nodes=<NODES>
#SBATCH --ntasks=<NTASKS>
#SBATCH --ntasks-per-node=<NTASKS_PER_NODE>
#SBATCH --ntasks-per-socket=<NTASKS_PER_SOCKET>
#SBATCH --cpus-per-task=<CPUS_PER_TASK>
#SBATCH --mem=<MEMORY>
#SBATCH --time=<TIME_LIMIT>
#SBATCH --partition=<PARTITION>
#SBATCH -A <ACCOUNT>

#SBATCH --output=./monte-carlo_%j.out
#SBATCH --error=./monte-carlo_%j.err

set -euo pipefail

module purge

# NOTE: Adapt the module names and versions below to match
# the software modules available on the target HPC cluster.
module load julia/1.12.4
module load openmpi/4.1.8/gnu

export PATH="$HOME/.julia/bin:$PATH"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../../../experiments" && pwd)"

MAIN_SCRIPT="$SCRIPT_DIR/../../../experiments/monte-carlo/torcjulia/main.jl"

THREADS_ARG="${SLURM_CPUS_PER_TASK},0" # Assuming Julia version >= 1.12.0, otherwise use "${SLURM_CPUS_PER_TASK}"

if [[ "$SLURM_NTASKS" -ne 1 ]]; then
    WORKER_THREADS=$((SLURM_CPUS_PER_TASK - 1))
else
    WORKER_THREADS="${SLURM_CPUS_PER_TASK}"
fi

NUM_RANKS="$SLURM_NTASKS"

TIME_OUTPUT=$(srun --mpi=pmix \
    --distribution=block:block \
    --cpu-bind=cores,verbose \
    julia -O3 \
    --project="$PROJECT_DIR" \
    --threads="$THREADS_ARG" \
    "$MAIN_SCRIPT" \
)

WALL_CLOCK_TIME=$(printf '%s\n' "$TIME_OUTPUT" | sed -n 's/.*Time taken[^:]*: *\([0-9.]*\) seconds.*/\1/p' | tail -n 1)

printf 'MPI Processes: %s, Worker Threads: %s | Wall-Clock Time: %s seconds\n' \
        "$NUM_RANKS" "$WORKER_THREADS" "$WALL_CLOCK_TIME"