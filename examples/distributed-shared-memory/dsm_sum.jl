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
Example: Parallel Array Sum with Distributed Shared Memory (ArgoDSM) in torcjulia

Description: This example demonstrates how the DSM abstraction can be used on top of torcjulia to compute the sum of an array in parallel. 
             The primary application task allocates and initializes the shared memory region `A`, which is accessible to tasks running on 
             different MPI nodes. Then, `N` tasks are created, where `N` is the number of MPI nodes used to launch torcjulia. Each task is 
             explicitly scheduled to a different MPI node and executes the `work` function, which computes the partial sum of its assigned 
             segment of `A` and writes the result to the corresponding position in another shared memory region `B`. Finally, the total sum 
             is computed by summing all elements in `B`.

Requirements: 2+ MPI processes, 1+ worker thread(s) per process.

Julia < v1.12:
Run: mpiexecjl -n 8 julia --project=/path/to/torcjulia/project --threads 2 dsm_sum.jl

Julia >= v1.12:
Run: mpiexecjl -n 8 julia --project=/path/to/torcjulia/project --threads 2,0 dsm_sum.jl
"""

using torcjulia

function work(A::Vector{Int}, B::Vector{Int})
    rank = torcjulia.node_id()
    partial_sum = Base.sum(A[4*rank+1:4*rank+4])

    B[rank + 1] = partial_sum
    torcjulia.argo_barrier(1)
end

function main()
    N = torcjulia.num_nodes()
    
    A = torcjulia.pdsm_alloc(4 * Int(N), Int)
    println("[MPI Rank: $(torcjulia.node_id())] Allocated ArgoDSM array A ($(4 * N) integer elements).")

    B = torcjulia.pdsm_alloc(Int(N), Int)
    println("[MPI Rank: $(torcjulia.node_id())] Allocated ArgoDSM array B ($N integer elements).")

    for i in 1:4*N A[i] = i end

    println("[MPI Rank: $(torcjulia.node_id())] Initial values of A: $A.")

    tasks = Vector{torcjulia.Future}(undef, N)
    
    for i in 0:N-1
        tasks[i+1] = torcjulia.submit(work, A, B; node = i)
    end
    
    torcjulia.wait(tasks)

    println("[MPI Rank: $(torcjulia.node_id())] Final values of B after all tasks completed: $B.")

    @assert B == collect(Int, 10:16:16*N-6)

    total_sum = Base.sum(B)

    println("[MPI Rank: $(torcjulia.node_id())] Total sum of elements in A: $total_sum.")

    @assert total_sum == Base.sum(10:16:16*N-6)
end

# Configure the ArgoDSM memory and cache capacities
torcjulia.argo_init(1024*1024, 1024*4)
torcjulia.init(main)