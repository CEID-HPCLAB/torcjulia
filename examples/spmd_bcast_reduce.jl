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
Example: SPMD Execution Model with Direct MPI Collective Operations (Calls) in torcjulia

Description: This example demonstrates combining torcjulia's SPMD execution model with direct MPI collective operations (calls).

Workflow:
    1. The root node (rank 0) initializes array A with values [0, 100, 200, ...].
    2. A is broadcast to all nodes via MPI.Bcast!.
    3. Each node scales its copy of A by its rank (A .*= rank).
    4. The scaled arrays are summed element-wise across all nodes via MPI.Allreduce.

Requirements: 2+ MPI processes, 1+ worker thread(s) per process.

Julia < v1.12
Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 2 spmd_bcast_reduce.jl

Julia >= v1.12
Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 2,0 spmd_bcast_reduce.jl
"""

using torcjulia

using MPI

nodes = torcjulia.num_nodes()

if nodes < 2
    println("[MPI Rank: $(torcjulia.node_id())] this example needs at least two MPI processes. exiting...")
    exit()
end

const N = 16
const A = zeros(Int, N)

function bcast_task()
    comm = MPI.COMM_WORLD
    MPI.Bcast!(A, comm)
end

function local_computation()
    node_id = torcjulia.node_id()
    
    A .*= node_id
    
    println("Node $node_id (after local computation): A = $A.")
end

function reduce_task()
    node_id = torcjulia.node_id()

    comm = MPI.COMM_WORLD
    result = MPI.Allreduce(A, +, comm)

    println("Node $node_id (after reduction): Element-wise sum of A = $result.")
end

function main()
    println("Running with $(torcjulia.num_nodes()) MPI nodes.")

    for i in 1:N
        A[i] = 100 * (i - 1)
    end
    
    println("Node 0 (initialization): A = $A.")

    torcjulia.spmd(bcast_task)
    torcjulia.spmd(local_computation)
    torcjulia.spmd(reduce_task)
    
    println("All nodes completed their assigned tasks successfully.")
end

torcjulia.init(main)