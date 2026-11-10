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
Example: Task Callbacks in torcjulia

Description: This example demonstrates the usage of task callbacks in torcjulia. Each task computes the square of an integer 
             and invokes a callback function, depending on the task's ID, to display the result.

Notes:
-  Callbacks receive the task result as input.
- `async_callback = false`: callbacks execute immediately after task completion.
- `async_callback = true`: callbacks are queued as separate tasks for asynchronous execution.
- In this illustrative example, tasks with even ids invoke `cb_1`, while tasks with odd ids invoke `cb_2`.

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3 callback.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3,0 callback.jl
"""

using torcjulia

function cb_1(result::Int)
    local_thread_id = torcjulia.worker_local_id()
    println("Thread $local_thread_id on node $(torcjulia.node_id()): Callback 1 Result = $result.")
end

function cb_2(result::Int)
    local_thread_id = torcjulia.worker_local_id()
    println("Thread $local_thread_id on node $(torcjulia.node_id()): Callback 2 Result = $result.")
end

function work(x::Int)::Int
    y = x^2
    
    local_thread_id = torcjulia.worker_local_id()
    println("Thread $local_thread_id on node $(torcjulia.node_id()): Work input = $x, output = $y.")
    
    y
end

function main()
    ntasks = 16
    
    data = 1:ntasks

    println("Submitting $ntasks tasks.")

    tasks = [
        torcjulia.submit(work, x; 
            callback = x % 2 == 0 ? cb_1 : cb_2, 
            async_callback = false
        ) for x in data
    ]
    
    torcjulia.wait(tasks)

    for (idx, task) in enumerate(tasks)
        res = torcjulia.result(task)
        println("Task $idx: $(torcjulia.input(task)[1])² = $res.")
    end

    println("All tasks completed successfully.")
end

torcjulia.init(main)