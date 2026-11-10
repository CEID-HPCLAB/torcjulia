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
 * Authors (CEID-HPCLAB): Argiris Sofotasios
 * License:               MIT - See LICENSE file in the project root
 * ==================================================================================
=#

"""
Example: Single Program Multiple Data (SPMD) Execution Model in torcjulia

Description: This example demonstrates torcjulia's SPMD execution model (spmd function), where the same task function runs on all MPI nodes. 

Workflow: Each MPI node executes the same function `work`, but with different data based on its MPI rank (node ID).

Requirements: 2+ MPI processes, 1+ worker thread(s) per process

Julia < v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 2 spmd.jl

Julia >= v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 2,0 spmd.jl
"""

using torcjulia

nodes = torcjulia.num_nodes()

if nodes < 2
    println("[MPI Rank: $(torcjulia.node_id())] this example needs at least two MPI processes. exiting...")
    exit()
end

function work(; multiplier::Int)::Tuple{Int, Int}
    node_id = torcjulia.node_id()
    
    A = [(node_id * 10 + i) * multiplier for i in 1:node_id]
    
    sum(A), node_id
end

function main()
    println("Running with $(torcjulia.num_nodes()) MPI nodes.")
    
    tasks = torcjulia.spmd(work; multiplier = 2)
    
    for (i, task) in enumerate(tasks)
        res = torcjulia.result(task)
        
        println("Node $(res[2]): sum(A) = $(res[1]).")
    end

    println("All nodes completed their assigned task successfully.")
end

torcjulia.init(main)