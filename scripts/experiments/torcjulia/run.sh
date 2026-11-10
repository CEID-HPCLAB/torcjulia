#!/usr/bin/env bash

# This script reproduces all results for all four task-parallel workloads
# using torcjulia as the underlying AMT runtime system.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

bash ${SCRIPT_DIR}/run_workload1.sh
bash ${SCRIPT_DIR}/run_workload2.sh
bash ${SCRIPT_DIR}/run_workload3.sh
bash ${SCRIPT_DIR}/run_workload4.sh

echo
echo "The experiments for all four workloads completed successfully."