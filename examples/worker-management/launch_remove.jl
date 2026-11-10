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
Example: Dynamic spawning and termination of MPI processes in torcjulia

Description: This example demonstrates dynamic process management in torcjulia by adding and removing MPI processes at runtime. 
             The master process (rank 0) initially submits 32 tasks, computing the squares of integers from 1 to 32. It then dynamically 
             adds four MPI processes and re-executes the same set of tasks. Subsequently, it removes two MPI processes one at a time, 
             executing the same tasks after each removal.

Notes:
- Each task introduces an artificial delay of 1.5 seconds to illustrate the impact of adding 
  and removing MPI processes on overall execution time.
- The `add_worker` function is used to dynamically add MPI processes, while the `remove_worker` function is used to remove them.
- WARNING: The initial processes of the MPI communicator cannot be removed.

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 2 launch_remove.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 2,0 launch_remove.jl
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
    ntasks = 32
    
    inputs = 1:ntasks

    t0 = time_ns() * 1e-9

    tasks = [torcjulia.submit(work, x) for x in inputs]
    
    torcjulia.wait(tasks)

    elapsed = time_ns() * 1e-9 - t0

    @assert all(torcjulia.result(task) == inputs[i]^2 for (i, task) in enumerate(tasks))

    println("Total elapsed time: $(round(elapsed, digits = 3)) seconds with $(torcjulia.num_nodes()) MPI processes \
                and $(torcjulia.num_local_workers()) Julia worker threads per process.")
    
    num_new_procs = 4

    torcjulia.add_worker(; num_procs = num_new_procs)

    println("Added $num_new_procs new MPI processes. Total MPI processes: $(torcjulia.num_nodes()).")

    t0 = time_ns() * 1e-9

    tasks = [torcjulia.submit(work, x) for x in inputs]
    
    torcjulia.wait(tasks)

    elapsed = time_ns() * 1e-9 - t0

    @assert all(torcjulia.result(task) == inputs[i]^2 for (i, task) in enumerate(tasks))

    println("Total elapsed time: $(round(elapsed, digits = 3)) seconds with $(torcjulia.num_nodes()) MPI processes \
                and $(torcjulia.num_local_workers()) Julia worker threads per process.")
    
    remove_node_id = 4
    
    torcjulia.remove_worker(; node_id = remove_node_id)

    println("Removed MPI process: $remove_node_id. Total MPI processes: $(torcjulia.num_nodes()).")

    t0 = time_ns() * 1e-9

    tasks = [torcjulia.submit(work, x) for x in inputs]
    
    torcjulia.wait(tasks)

    elapsed = time_ns() * 1e-9 - t0

    @assert all(torcjulia.result(task) == inputs[i]^2 for (i, task) in enumerate(tasks))

    println("Total elapsed time: $(round(elapsed, digits = 3)) seconds with $(torcjulia.num_nodes()) MPI processes \
                and $(torcjulia.num_local_workers()) Julia worker threads per process.")
    
    remove_node_id = 2

    torcjulia.remove_worker(; node_id = remove_node_id)

    println("Removed MPI process: $remove_node_id. Total MPI processes: $(torcjulia.num_nodes()).")
    
    t0 = time_ns() * 1e-9

    tasks = [torcjulia.submit(work, x) for x in inputs]

    torcjulia.wait(tasks)

    elapsed = time_ns() * 1e-9 - t0

    @assert all(torcjulia.result(task) == inputs[i]^2 for (i, task) in enumerate(tasks))

    println("Total elapsed time: $(round(elapsed, digits = 3)) seconds with $(torcjulia.num_nodes()) MPI processes \
                and $(torcjulia.num_local_workers()) Julia worker threads per process.")
end

torcjulia.init(main)