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
Example: MPI Interoperability in torcjulia

Description: This example demonstrates the interoperability of torcjulia with MPI. It showcases how developers can seamlessly 
             integrate the AMT parallelism model with direct MPI (collective) operations.

The example includes:
- MPI operations BEFORE torcjulia's runtime engine initialization (MPI.gather, MPI.Allreduce).
- MPI operations INSIDE torcjulia tasks (same operations as outside, but within the context of torcjulia tasks).
- multiple torcjulia.init() calls with MPI_finalize control.

Notes:
- `MPI_finalize = false`: Keeps MPI alive between torcjulia sessions.
- Direct MPI operations coexist with torcjulia task-based execution.

Requirements: 2+ MPI processes, 1+ worker thread(s) per process.

Julia < v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 2 in_out_torcjulia_mpi.jl

Julia >= v1.12:
Run: mpiexecjl -n 4 julia --project=/path/to/torcjulia/project --threads 2,0 in_out_torcjulia_mpi.jl
"""

using torcjulia

nodes = torcjulia.num_nodes()

if nodes < 2
    println("[Node: $(torcjulia.node_id())] this example needs at least two MPI processes. exiting...")
    exit()
end

using MPI

rank = MPI.Comm_rank(MPI.COMM_WORLD)   

function mpi_gather_outside()
    local_val = [rank + 1]
    all_vals = MPI.gather(local_val, MPI.COMM_WORLD)
    
    if rank == 0
        println("Outside torcjulia (gather) $(vcat(all_vals...)).")
    end
end

function mpi_gather_inside()
    user_comm = MPI.COMM_WORLD
    local_val = [rank + 1]
    all_vals = MPI.gather(local_val, user_comm)

    if rank == 0
        println("Inside torcjulia (gather) $(vcat(all_vals...)).")
    end
end

function torcjulia_gather()
    torcjulia.spmd(mpi_gather_inside)
end

function mpi_allreduce_outside()
    local_val = rank + 1
    total_sum = MPI.Allreduce(local_val, +, MPI.COMM_WORLD)
    
    if rank == 0
        println("Outside torcjulia (Allreduce) $total_sum.")
    end
end

function mpi_allreduce_inside()
    user_comm = MPI.COMM_WORLD
    local_val = rank + 1
    total_sum = MPI.Allreduce(local_val, +, user_comm)
    
    if rank == 0
        println("Inside torcjulia (Allreduce) $total_sum.")
    end
end

function torcjulia_allreduce()
    torcjulia.spmd(mpi_allreduce_inside)
end

mpi_gather_outside()

torcjulia.init(torcjulia_gather; MPI_finalize = false)

mpi_allreduce_outside()

torcjulia.init(torcjulia_allreduce)