#!/usr/bin/env bash

# This script generates the synthetic dataset (126 GB, 701 features: x0, x1, x2, ..., x699, y)
# for the third workload (Parallel Preprocessing of a Large Tabular Dataset).
# The generated dataset is saved in experiments/data-preprocessing/data/dataset.csv.
#
# It also generates a smaller synthetic dataset (700 MB) with the same 701 features,
# which is used during the warmup phase of the third workload.
# The warmup dataset is saved in experiments/data-preprocessing/data/warmup_dataset.csv.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../../experiments/data-preprocessing/data"

mkdir -p "$DATA_DIR"

python3 create_dataset.py --samples 100_000 --chunks 10 --path "$DATA_DIR" --features 700 --classes 4 -pbn "warmup_dataset"

python3 create_dataset.py --samples 17_978_709 --chunks 10 --path "$DATA_DIR" --features 700 --classes 4 -pbn "dataset"