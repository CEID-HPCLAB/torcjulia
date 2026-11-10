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
Example: Task Pinning in torcjulia

Description: This example demonstrates task pinning in torcjulia. Tasks can be pinned to specific MPI nodes, 
             ensuring they execute on the designated node. This is useful for scenarios where data locality or 
             resource affinity is important. 
             
Notes:
- The `node` argument controls which node a task is assigned to.
- When work stealing is enabled, pinned tasks can still be stolen by idle nodes unless `forced_node = true` is set, which guarantees the task 
  executes exclusively on the assigned node.

Three cases (all tasks are submitted to node 0):

Case 1 - Pinned to node 0, stealing disabled:
  Tasks remain on node 0, other nodes remain idle.

Case 2 - Pinned to node 0, stealing enabled:
  Other nodes can steal pending tasks from node 0.

Case 3 - Pinned to node 0, stealing enabled, forced_node = true:
  Tasks are guaranteed to execute on node 0, even though work stealing is enabled. 

Requirements: 2+ MPI processes, 1 worker thread per process.

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 2 task_pinning.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 2,0 task_pinning.jl
"""

using torcjulia

nodes = torcjulia.num_nodes()

if nodes < 2
    println("[MPI Rank: $(torcjulia.node_id())] this example needs at least two MPI processes. exiting...")
    exit()
end

local_workers = torcjulia.num_local_workers()

if local_workers > 1
    println("[MPI Rank: $(torcjulia.node_id())] this example should use one Julia worker thread per MPI process. exiting...")
    exit()
end

function work(x::Int)::Int
    torcjulia.node_id() % 2 == 0 ? sleep(0.2) : nothing
    
    y = x * x
    
    println("Node $(torcjulia.node_id()) (worker thread $(torcjulia.worker_local_id())): work (input = $x, output = $y).")

    y
end

function main()
    ntasks = 10
    
    println("Scenario 1: Pinned to node 0, work stealing disabled (default).")

    tasks = [torcjulia.submit(work, i; node = 0) for i in 1:ntasks]
    torcjulia.wait(tasks)

    torcjulia.enable_stealing()
    println("\nScenario 2: Pinned to node 0, work stealing enabled and forced_node = false (tasks can be stolen).")

    tasks = [torcjulia.submit(work, i; node = 0) for i in 1:ntasks]
    torcjulia.wait(tasks)

    println("\nScenario 3: Pinned to node 0, work stealing enabled and forced_node = true (tasks cannot be stolen).")

    tasks = [torcjulia.submit(work, i; node = 0, forced_node = true) for i in 1:ntasks]
    torcjulia.wait(tasks)
end

torcjulia.init(main)