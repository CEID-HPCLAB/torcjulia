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
Example: Local Scheduling Policy in torcjulia

Description: This example demonstrates torcjulia's local scheduling policy, where tasks are assigned to the same MPI node that submitted them.

Note: The local scheduling policy is set via `set_scheduling_policy(:LOCAL)`. 

In this example, 12 tasks are submitted to compute the square of integers from 1 to 12. 
Since all tasks are submitted by the master node (rank 0), all tasks are assigned to node 0, and other nodes remain idle.
If work stealing were enabled (disabled by default), tasks could be stolen and executed on remote nodes, but this is not the case in this example.

Julia < v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 3 local.jl

Julia >= v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 3,0 local.jl
"""

using torcjulia

function work(x::Int)::Int
    y = x^2
    
    println("Work: Input = $x, Output = $y \
             on worker thread $(torcjulia.worker_local_id()) of node $(torcjulia.node_id()).")

    y
end

function main()
    ntasks = 12
    
    inputs = 1:ntasks

    torcjulia.set_scheduling_policy(:LOCAL)

    tasks = [torcjulia.submit(work, x) for x in inputs]
    torcjulia.wait(tasks)

    for (i, task) in enumerate(tasks)
        x = inputs[i]
        
        y = torcjulia.result(task)
        
        println("Task $i: $(x)² = $y.")
    end
end

torcjulia.init(main)