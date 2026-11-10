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
Example: Parallel MapPairs Algorithmic Skeleton

Description: Demonstrates the usage of the parallel `mapPairs` algorithmic skeleton in torcjulia.

Notes: The parallel `mapPairs` skeleton applies a user-defined function to every (x, y) pair from the Cartesian 
       product of two input arrays. In this example, we define a simple function `work` that computes the product of two integers.

Illustrative example: `mapPairs(work, [1, 2], [10, 20])` → `[10 20; 20 40]`

Key parameters:
- `work`: A user-defined function that takes two integer arguments and returns their product.
- `a`, `b`: Two input arrays of integers, each containing N elements (1,000 in this case).
- `chunksize`: The number of pairs each task will process (5,000 in this case). If chunksize is not specified, 
               the AMT runtime system will automatically

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 5 map_pairs.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 5,0 map_pairs.jl
"""

using torcjulia

work(x::Int, y::Int)::Int = x * y

function main()
    N = 1_000; chunksize = 5_000

    a = 1:N; b = N:-1:1
    
    println("Each task executes the work function for $chunksize pairs.")
    println("Expected number of tasks = $(ceil(Int, (length(a) * length(b)) / chunksize)).")

    results = torcjulia.mapPairs(work, a, b; chunksize = chunksize)
    
    @assert results == [work(x, y) for x in a, y in b]  
end

torcjulia.init(main)