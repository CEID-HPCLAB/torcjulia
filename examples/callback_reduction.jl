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
Example: Task Callbacks for Reduction in torcjulia

Description: This example demonstrates the usage of callbacks in torcjulia to perform a reduction operation. 
             Each task computes the square of an integer and invokes a callback function to accumulate the result into a global sum.

Notes: 
- In torcjulia, callbacks are always executed on the workers of the node that created the task (the homenode). Since all tasks are 
  created by rank 0, all callbacks run on rank 0, ensuring safe access to the global `sum` variable.
- If work stealing were enabled (disabled by default), callback tasks could be stolen and executed on remote nodes
  where the global `sum` variable is not shared.
- The global variable `sum` should be protected with a lock/semaphore for thread safety when 
  multiple worker threads execute callbacks concurrently.

Julia < v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 3 callback_reduction.jl

Julia >= v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 3,0 callback_reduction.jl
"""

using torcjulia

global_sum = 0
sum_lock = ReentrantLock()  

function work(x::Int)::Int
    x^2
end

function cb(result::Int)
    global global_sum

    lock(sum_lock) do
        global_sum += result
    end
end

function main()
    ntasks = 20
    
    data = 1:ntasks
    
    println("Computing sum of squares from 1 to $ntasks using callbacks.")
    
    tasks = [torcjulia.submit(work, x; callback = cb, async_callback = false) for x in data]
    torcjulia.wait(tasks)

    println("Sum of squares from 1 to $ntasks: $global_sum.")
    println("Expected: $(sum(x^2 for x in 1:ntasks)).")

    @assert global_sum == sum(x^2 for x in 1:ntasks)
    
    println("All tasks completed successfully.")
end

torcjulia.init(main)