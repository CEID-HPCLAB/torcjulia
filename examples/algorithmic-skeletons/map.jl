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
Example: Parallel Map Algorithmic Skeleton

Description: Demonstrates the usage of the parallel `map` algorithmic skeleton in torcjulia.

Notes: The parallel `map` skeleton applies a user-defined function to each element of the input iterable. 
       In this example, we define a simple function `work` that computes the square of a given integer.

Illustrative example: `map(work, [1, 2, 3, 4, 5])` → `[1, 4, 9, 16, 25]`

Key parameters:
- `work`: A user-defined function that takes an integer and returns its square.
- `data`: An array of integers from 1 to N (10,000 in this case).
- `chunksize`: The number of elements each task will process (1,000 in this case). If chunksize is not specified, 
               the AMT runtime system will automatically determine the optimal chunk size based on the number of available workers.

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3 map.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3,0 map.jl
"""
    
using torcjulia

work(x::Int)::Int = x^2

function main()
    N = 10_000; chunksize = 1_000
    
    data = 1:N

    println("Each task computes the square of $chunksize elements.")
    println("Expected number of tasks = $(ceil(Int, N / chunksize)).")
    
    results = torcjulia.map(work, data; chunksize = chunksize)

    @assert results == [x^2 for x in data]
end

torcjulia.init(main)