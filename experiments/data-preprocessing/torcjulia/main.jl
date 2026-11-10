#=
 * ==================================================================================
 * torcjulia — An MPI-based Runtime System for Asynchronous Task Parallelism in Julia
 * ==================================================================================
 * This file is part of torcjulia, an MPI-based runtime system for asynchronous
 * many task (AMT) parallelism in Julia across both shared and distributed memory
 * platforms, developed at the High Performance Computing Laboratory (HPCLAB)
 * at the Computer Engineering & Informatics Department (CEID),
 * University of Patras, Greece.
 *
 * Repository: https://github.com/CEID-HPCLAB/torcjulia
 *
 * Authors: Argiris Sofotasios (CEID-HPCLAB)
 * License: MIT License — see the LICENSE file in the repository root.
 * ==================================================================================
=#

"""
Experiment: Parallel Preprocessing of a Large Tabular Dataset using torcjulia (Workload 3)

Description: Parallel preprocessing of a large tabular dataset stored in CSV format using torcjulia, following the 
             task-based parallelism scheme proposed in [1]. The dataset consists of 701 features with a total size of 126 GB. 
             All features are numerical except the last one, which is a categorical target variable with four distinct classes. 
             
             The preprocessing scenario involves fitting a StandardScaler (Z-score normalization) to the dataset's input features. 
             The dataset is divided into 600 equally-sized chunks, each processed by an individual task.
             
Notes:
    - The experiment uses the dataset specified by the `DATASET_FILEPATH` environment variable.
    - The execution times reported in the manuscript were obtained using a synthetic 126 GB tabular dataset.
    - To generate this dataset, run `create.sh` from `scripts/data/`. 
      The script is configured to generate the required dataset and save it in `experiments/data-preprocessing/data/`.
    - The experiment also includes a warm-up run using a smaller toy dataset (100,000 samples) to exclude compilation overhead from 
      the fitting phase's execution time. The warm-up dataset is specified by the `WARMUP_DATASET_FILEPATH`` environment variable and can 
      be generated using the same create.sh script by setting the `--samples` argument to 100_000 and the --pbn argument to `warmup_dataset`.
    - If the binary block offsets are not available, Mneme.BlockReader computes them automatically and saves them to the
      file specified by the `OFFSETS_FILEPATH` environment variable for use in subsequent runs.

References:
[1] Argiris Sofotasios, Dimitris Metaxakis, and Panagiotis Hadjidoukas. Mneme: A Parallel Preprocessing Framework for Large Tabular Datasets. 
    International Journal of Parallel Programming 54, 20 (2026). doi:10.1007/s10766-026-00817-7
"""

using torcjulia

include(joinpath(@__DIR__, "src", "mneme.jl"))
import .Mneme

using Printf

const DATASET_FILEPATH = Base.get(ENV, "DATASET_FILEPATH", "/path/to/file/dataset.csv") # Path to the input CSV dataset file
const OFFSETS_FILEPATH = Base.get(ENV, "OFFSETS_FILEPATH", "/path/to/file/offsets.dat") # Path to the file containing precomputed block offsets (if available)

const NROWS = 17_978_709 # Total number of dataset's samples
const NCOLS = 700 # Total number of dataset's input features
const NUM_BLOCKS = 600 # Number of blocks to split the dataset into

const WARMUP_DATASET_FILEPATH = Base.get(ENV, "WARMUP_DATASET_FILEPATH", "/path/to/file/warmup_dataset.csv") # Path to the toy warmup CSV dataset file
const WARMUP_OFFSETS_FILEPATH = Base.get(ENV, "WARMUP_OFFSETS_FILEPATH", "/path/to/file/warmup_offsets.dat") # Path to the file containing precomputed block offsets for the warmup dataset (if available)

const WARMUP_NROWS = 100_000 # Total number of warmup dataset's samples
const WARMUP_NCOLS = 700 # Total number of warmup dataset's input features
const WARMUP_NUM_BLOCKS = 50 # Number of blocks to split the warmup dataset into

function main()
    warmup_reader = Mneme.BlockReader(WARMUP_DATASET_FILEPATH; num_blocks = WARMUP_NUM_BLOCKS, 
                               num_rows = WARMUP_NROWS, offsets_path = WARMUP_OFFSETS_FILEPATH)
                               
    # Warmup run to avoid compilation overhead in the timing of the fitting phase
    standard_scaler_warmup = Mneme.StandardScaler(WARMUP_DATASET_FILEPATH, warmup_reader.columns[1:WARMUP_NCOLS])
    Mneme.fit(standard_scaler_warmup, warmup_reader)

    reader = Mneme.BlockReader(DATASET_FILEPATH; num_blocks = NUM_BLOCKS, 
                               num_rows = NROWS, offsets_path = OFFSETS_FILEPATH)
    
    # Z-score normalization on the dataset's input features
    standard_scaler = Mneme.StandardScaler(DATASET_FILEPATH, reader.columns[1:NCOLS])

    t0 = torcjulia.gettime()

    # Compute the mean and standard deviation for each specified feature (Fitting Phase)
    Mneme.fit(standard_scaler, reader)

    elapsed = torcjulia.gettime() - t0

    @printf("Time taken to fit the StandardScaler (Z-score Normalization): %.4f seconds\n", elapsed)
    
    # Print the computed statistics for each specified feature
    # Mneme.print_stats(standard_scaler)
end

torcjulia.init(main)