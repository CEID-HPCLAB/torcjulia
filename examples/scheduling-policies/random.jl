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
Example: Random Scheduling Policy in torcjulia

Description: This example demonstrates torcjulia's random scheduling policy, where tasks are assigned to MPI nodes in a non-deterministic manner.

Note: The random scheduling policy is set via `set_scheduling_policy(:RANDOM)`.

Julia < v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 3 random.jl

Julia >= v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 3,0 random.jl
"""

using torcjulia

function work(x::Int)::Int
    y = x^2
    
    println("Work: Input = $x, Output = $y \
             on worker thread $(torcjulia.worker_local_id()) of node $(torcjulia.node_id()).")

    y
end

function main()
    ntasks = 14
    
    inputs = 1:ntasks

    torcjulia.set_scheduling_policy(:RANDOM)

    tasks = [torcjulia.submit(work, x) for x in inputs]
    torcjulia.wait(tasks)

    for (i, task) in enumerate(tasks)
        x = inputs[i]
        
        y = torcjulia.result(task)
        
        println("Task $i: $(x)² = $y.")
    end
end

torcjulia.init(main)