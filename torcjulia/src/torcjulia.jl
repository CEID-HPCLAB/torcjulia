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

module torcjulia

include("./runtime.jl")
import .runtime: add_worker, argo_barrier, argo_init, ctx, filter, finalize, get, gettime, init, input,
                 kw_input, launch, map, mapReduce, mapPairs, mapPairsReduce,
                 map_async, pdsm_alloc, ready, reduce, remove_worker, result, 
                 scan, set_logger_level, shw_alloc, spmd, starmap,
                 starmap_async, submit, wait,
                 DistributionPolicy, Future, TaskPriority, TaskPriorityLevel

export add_worker, argo_barrier, argo_init, filter, get, gettime, init, input, kw_input,
       map, mapReduce, mapPairs, mapPairsReduce, map_async, pdsm_alloc, ready, 
       reduce, remove_worker, result, scan, set_logger_level,
       shw_alloc, spmd, starmap, starmap_async, submit, submit_wait,
       Future, TaskPriority, TaskPriorityLevel

# MPI rank of the calling process (0-based)
const node_id = () -> Int(runtime.ctx[].rank) 

# Total number of MPI processes in torcjulia's runtime engine
const num_nodes = () -> Int(runtime.ctx[].num_procs) 

# Number of Julia worker threads per MPI process
const num_local_workers = () -> Int(runtime.TORCJULIA_NUM_WORKERS[]) 

# Total number of Julia worker threads across all MPI processes
const num_workers = () -> Int(runtime.ctx[].num_procs) * Int(runtime.TORCJULIA_NUM_WORKERS[])

# ID of the calling Julia worker thread within its MPI process (1-based) (local ID)
const worker_local_id = () -> Int(runtime._torcjulia_tls_get_id()) 

# ID of the calling Julia worker thread across all MPI processes (1-based) (global ID)
const worker_id = () -> Int(runtime.ctx[].rank) * Int(runtime.TORCJULIA_NUM_WORKERS[]) + Int(runtime._torcjulia_tls_get_id()) 

enable_stealing() = (runtime._enable_stealing()) # Enables the work stealing mechanism (Default: disabled)
disable_stealing() = (runtime._disable_stealing()) # Disables the work stealing mechanism
is_stealing_enabled()::Bool = (runtime._get_stealing_status()) # Returns the current status of the work stealing mechanism (true if enabled, false otherwise)

"""
    set_scheduling_policy(policy::Symbol, weights::Union{Vector{Int}, Nothing} = nothing)

Sets the task scheduling policy used by the torcjulia runtime environment.

The `policy` argument specifies the scheduling policy and must be one of the following symbols:
- `:ROUND_ROBIN`: Distributes tasks across MPI nodes in a round-robin (cyclic) fashion.
- `:WRR`: Distributes tasks according to the specified weights, with one weight per MPI node.
- `:LOCAL`: Assigns each task to the MPI node that submitted it.
- `:RANDOM`: Assigns each task to a randomly selected MPI node.

The `weights` argument is used only when `policy` is `:WRR`. It must be a vector of integers, with each element specifying the weight of the
corresponding MPI node. For example, with four MPI nodes and weights `[1, 1, 2, 3]`, node 0 and node 1 will receive tasks in a 1:1 ratio, 
node 2 will receive tasks in a 2:1 ratio compared to nodes 0 and 1, and node 3 will receive tasks in a 3:1 ratio compared to nodes 0 and 1. 

For all other policies, `weights` is ignored.
"""
set_scheduling_policy(policy::Symbol, weights::Union{Vector{Int}, Nothing} = nothing) = (runtime._set_scheduling_policy(policy, weights))

get_scheduling_policy()::DistributionPolicy = (runtime._get_scheduling_policy())

"""
    submit_wait(func, args...; kwargs...)

Encapsulates the submit-wait pattern by submitting a task to execute `func` 
with the provided arguments, waiting for it to complete, and returning its result.
"""
function submit_wait(func, args...; kwargs...)
    task = submit(func, args...; kwargs...)
    
    wait(task)
    
    result(task)
end

"""
    init(func::Function; MPI_finalize::Bool = true)

Initializes the torcjulia AMT runtime system and launches the provided function `func` on the main thread of the master 
MPI process (rank 0) as the primary application task. Once `func` completes, the runtime system is shut down.
The MPI environment is also finalized if `MPI_finalize` is `true` (default). If `MPI_finalize` is set to `false`, the MPI 
environment remains active, allowing further MPI operations after `func` completes. 
"""
function init(func::Function; MPI_finalize::Bool = true)
    init()  

    launch(func) 
    
    finalize(MPI_finalize) 
end

end