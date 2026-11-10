#!/usr/bin/env bash

# This script reproduces all results for the two workloads with irregular task granularity
# using Dagger as the underlying AMT runtime system.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

bash ${SCRIPT_DIR}/run_workload1.sh
bash ${SCRIPT_DIR}/run_workload4.sh

echo
echo "The experiments for the two workloads with irregular task granularity completed successfully."