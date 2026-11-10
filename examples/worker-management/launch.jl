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
Example: Dynamically Launching MPI Processes in torcjulia

Description: This example demonstrates how to dynamically launch additional MPI processes in torcjulia. 
             The master process (MPI rank 0) submits 20 tasks to compute the squares of the integers from 1 to 20 
             and waits for all tasks to complete. It then dynamically adds two MPI processes, resubmits the same 
             set of tasks, and waits for their completion.

Notes: Each task introduces an artificial delay of 1.5 seconds to illustrate the impact of adding MPI processes on overall execution time.

Warning: Removing an MPI node automatically updates the ranks of the remaining processes.
         For example, if four processes have MPI ranks 0, 1, 2, and 3, and the process with rank 2 is removed, 
         the remaining processes are assigned ranks 0, 1, and 2 (the process previously assigned rank 3 is reassigned rank 2).

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 2 launch.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 2,0 launch.jl
"""

using torcjulia

function work(x::Int)::Int
    sleep(1.5)
    
    y = x^2

    println("Work: input = $x, output = $y \
             on worker thread $(torcjulia.worker_local_id()) of node $(torcjulia.node_id()).")
    y
end

function main()
    ntasks = 20
    
    inputs = 1:ntasks

    t0 = time_ns() * 1e-9

    tasks = [torcjulia.submit(work, x) for x in inputs]
    
    torcjulia.wait(tasks)

    elapsed = time_ns() * 1e-9 - t0

   for (i, task) in enumerate(tasks)
        y = torcjulia.result(task)
        println("Task $i: $(inputs[i])² = $y.")
    end

    @assert all(torcjulia.result(task) == inputs[i]^2 for (i, task) in enumerate(tasks))

    println("Total elapsed time: $(round(elapsed, digits = 3)) seconds with $(torcjulia.num_nodes()) MPI processes \
                and $(torcjulia.num_local_workers()) Julia worker threads per process.")

    num_new_procs = 2

    torcjulia.add_worker(; num_procs = num_new_procs)

    println("Added $num_new_procs new MPI processes. Total MPI processes: $(torcjulia.num_nodes()).")
    
    t0 = time_ns() * 1e-9

    tasks = [torcjulia.submit(work, x) for x in inputs]
    
    torcjulia.wait(tasks)

    elapsed = time_ns() * 1e-9 - t0

    for (i, task) in enumerate(tasks)
        y = torcjulia.result(task)
        println("Task $i: $(inputs[i])² = $y.")
    end

    @assert all(torcjulia.result(task) == inputs[i]^2 for (i, task) in enumerate(tasks))

    println("Total elapsed time: $(round(elapsed, digits = 3)) seconds with $(torcjulia.num_nodes()) MPI processes \
             and $(torcjulia.num_local_workers()) Julia worker threads per process.")
end

torcjulia.init(main)