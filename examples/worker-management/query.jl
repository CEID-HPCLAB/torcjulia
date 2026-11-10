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
Example: Querying Worker Information in torcjulia

Description: This example demonstrates how to query the number of MPI nodes, the total number of Julia worker threads, 
             and the number of worker threads per node in torcjulia. Additionally, it shows how to retrieve the global 
             and local IDs of Julia worker threads. 

Notes: In this example, each worker thread displays its global and local IDs, along with the MPI rank of the node it belongs to.
       For clarity, output is color-coded by MPI rank.

Warning: This example supports up to 4 MPI processes.

Julia < v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 5 query.jl

Julia >= v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 5,0 query.jl
"""

using torcjulia

using Crayons

COLORS = Dict(
    0 => :red,
    1 => :green,
    2 => :cyan,
    3 => :yellow
)

function work()
    node_id = torcjulia.node_id()
    
    global_worker_id = torcjulia.worker_id()
    local_worker_id = torcjulia.worker_local_id()
    
    println(Crayon(foreground = COLORS[node_id]), "[MPI Rank: $node_id] Worker thread global ID: $global_worker_id, Worker thread local ID: $local_worker_id.")
end

function main()
    nodes = torcjulia.num_nodes()
    node_id = torcjulia.node_id()

    total_workers = torcjulia.num_workers()
    workers_per_node = torcjulia.num_local_workers()
    
    global_worker_id = torcjulia.worker_id()
    local_worker_id = torcjulia.worker_local_id()
    
    println("[MPI Rank: $node_id] Number of MPI nodes: $nodes.")
    println("[MPI Rank: $node_id] Total number of Julia worker threads: $total_workers.")
    println("[MPI Rank: $node_id] Number of Julia worker threads per node: $workers_per_node.")
    println("[MPI Rank: $node_id] Worker thread global ID (Main Thread): $global_worker_id.")
    println("[MPI Rank: $node_id] Worker thread local ID (Main Thread): $local_worker_id.\n")

    tasks = [torcjulia.submit(work; node = (i - 1) % nodes) for i in 1:total_workers]
    
    torcjulia.wait(tasks)
end

torcjulia.init(main)