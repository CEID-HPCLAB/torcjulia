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
Example: Parallel Map Algorithmic Skeleton with Multiple Input Iterables

Description: Demonstrates the usage of the parallel `map` algorithmic skeleton in torcjulia with multiple input iterables.

Notes: When multiple input iterables are provided, the parallel map skeleton applies a user-defined function to the elements 
       at matching positions across the input iterables. In this example, the `work` function computes the product of two floating-point numbers.

Illustrative example: `map(work, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])` → `[4.0, 10.0, 18.0]`

Key parameters:
- `work`: A user-defined function that takes two floating-point numbers and returns their product.
- `a`, `b`: Two arrays of floating-point numbers, each containing N elements (10,000 in this case).
- `chunksize`: The number of elements each task will process (1,000 in this case). If chunksize is not specified, 
               the AMT runtime system will automatically determine the optimal chunk size based on the number of available workers.

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3 map2.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3,0 map2.jl
"""

using torcjulia

work(a::Float64, b::Float64)::Float64 = a * b

function main()
    N = 10_000; chunksize = 1_000
    
    a = range(0, 2π, length = N); b = range(π, 3π, length = N)

    println("Each task computes the product of $chunksize elements from two input iterables.")
    println("Expected number of tasks = $(ceil(Int, N / chunksize)).")

    results = torcjulia.map(work, a, b; chunksize = chunksize)

    @assert results == [x * y for (x, y) in zip(a, b)]
end

torcjulia.init(main)