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

module config

export TORCJULIA_MANAGER_TAG, TORCJULIA_STEAL_RESPONSE_TAG, TORCJULIA_TASK_YIELDTIME, TORCJULIA_QUEUE_LEVELS

# MPI message tag used for communication with the manager thread on each MPI node
const TORCJULIA_MANAGER_TAG::UInt = 100 

const TORCJULIA_STEAL_RESPONSE_TAG::UInt = 101

# Delay before a worker thread retries task retrieval in the dispatch loop
const TORCJULIA_TASK_YIELDTIME::Float64 = 0.0001

# Maximum nesting depth for tasks. Tasks spawned beyond this depth are assigned to the maximum level
const TORCJULIA_QUEUE_LEVELS::UInt = 10

end