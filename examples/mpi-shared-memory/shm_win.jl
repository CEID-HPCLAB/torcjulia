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
Example: Shared Memory with MPI Windows in torcjulia

Description: This example demonstrates how MPI shared-memory windows can be used on top of torcjulia, enabling MPI processes
             within a shared memory island to directly map memory regions into a global shared address space. This paradigm can 
             reduce memory consumption by avoiding data replication across tasks running on different MPI processes.

Notes: In torjulia, shared-memory windows are allocated through the `shw_alloc` function, while the underlying MPI-3 RMA operations
       are handled internally by the runtime. In this example, a shared array (`A`) of size equal to the number of MPI processes is allocated, 
       and each process updates its corresponding element in the shared array.

Requirements: 2+ MPI processes, 1+ worker thread(s) per process.

Julia < v1.12:
Run: mpiexecjl -n 8 julia --project=/path/to/torcjulia/project --threads 2 shm_win.jl

Julia >= v1.12:
Run: mpiexecjl -n 8 julia --project=/path/to/torcjulia/project --threads 2,0 shm_win.jl
"""

using torcjulia

nodes = torcjulia.num_nodes()

if nodes < 2
    println("[MPI Rank: $(torcjulia.node_id())] this example needs at least two MPI processes. exiting...")
    exit()
end

function work(idx::Int; A::Vector{Int})
    old_value = A[idx]
    
    A[idx] += 5 * idx

    new_value = A[idx]
    
    println("[MPI Rank: $(torcjulia.node_id())] Worker thread $(torcjulia.worker_local_id()) of node $(torcjulia.node_id()) updated A[$idx] from $old_value to $new_value.")
end

function main()
    N = Int(nodes)
    
    A = torcjulia.shw_alloc((N,), Int)

    println("[MPI Rank: $(torcjulia.node_id())] Allocated MPI shared-memory array A ($N integer elements).")

    if torcjulia.node_id() == 0
        for i in 1:N
            A[i] = 100 * (i - 1)
        end
    end

    println("[MPI Rank: $(torcjulia.node_id())] Initial values of A: $A.")

    tasks = [torcjulia.submit(work, i; A = A, node = i-1) for i in 1:Int(N)]

    torcjulia.wait(tasks)

    println("[MPI Rank: $(torcjulia.node_id())] Final values of A after all tasks completed: $A.")

    @assert A == [100 * (i - 1) + 5 * i for i in 1:Int(N)]
end

torcjulia.init(main)