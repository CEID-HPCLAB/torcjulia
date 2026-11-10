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
Example: Parallel Filter Algorithmic Skeleton

Description: Demonstrates the usage of the parallel `filter` algorithmic skeleton in torcjulia.

Notes: The parallel `filter` skeleton applies a user-defined function to each element of the input iterable 
       and returns only those elements for which the function returns true. In this example, we define a 
       simple function `work` that checks if a number is even.

Illustrative example: `filter(work, [1, 2, 3, 4, 5, 6])` → `[2, 4, 6]`

Key parameters:
- `work`: A user-defined function that takes an integer and returns a boolean indicating whether the number is even.
- `data`: An array of integers from 1 to N (10,000 in this case).
- `chunksize`: The number of elements each task will process (1,000 in this case). If chnksize is not specified, 
               the AMT runtime system will automatically determine the optimal chunk size based on the number of available workers.

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3 filter.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3,0 filter.jl
"""

using torcjulia

work(x::Int)::Bool = x % 2 == 0

function main()
    N = 10_000; chunksize = 1_000
    
    data = 1:N

    println("Each task filters $chunksize elements.")
    println("Expected number of tasks = $(ceil(Int, N / chunksize)).")
    
    results = torcjulia.filter(work, data; chunksize = chunksize)

    @assert results == [x for x in data if x % 2 == 0]
end

torcjulia.init(main)