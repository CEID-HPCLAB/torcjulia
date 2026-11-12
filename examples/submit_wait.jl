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
Example: Submit-Wait Pattern in torcjulia

Description: This example demonstrates the submit-wait pattern in torcjulia. The master process (rank 0) submits 10 tasks to compute 
             the cube of integers from 1 to 10, then waits for their completion. While waiting, the master's main thread 
             actively participates in executing tasks rather than remaining idle. 

Notes: The example also demonstrates `submit_wait()`, which submits a task, waits for it to complete, and returns its result.

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 2 submit_wait.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 2,0 submit_wait.jl
"""

using torcjulia

function work(x::Int)::Int
    x^3
end

function main()
    ntasks = 10
    
    data = 1:ntasks
    
    println("Submitting $ntasks tasks.")
    
    tasks = [torcjulia.submit(work, x) for x in data]
    
    torcjulia.wait(tasks)

    for (i, task) in enumerate(tasks)
        res = torcjulia.result(task)
        println("Task $i: $(data[i])³ = $res.")
    end

    j = 11
    
    println("Submitting a task and waiting for it to complete using submit-wait.")
     
    res = torcjulia.submit_wait(work, j)

    println("Task $j: $(j)³ = $res.")

    println("All tasks completed successfully.")
end

torcjulia.init(main)