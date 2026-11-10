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
Example: Parallel MapReduce Algorithmic Skeleton

Description: Demonstrates the usage of the parallel `mapReduce` algorithmic skeleton in torcjulia.

Notes: The parallel `mapReduce` skeleton combines the functionalities of the `map` and `reduce` algorithmic skeletons. 
       It first applies a user-defined function to each element of the input iterable (the "map" phase) 
       and then aggregates the results using a user-defined associative operation (the "reduce" phase). 
       In this example, we define a simple function `work` that computes the square of a given integer, 
       and a reduce function `aggregate` that sums two integers.

Illustrative example: `mapReduce(map_func, reduce_func, data)` → `reduce_func(map_func(data[1]), map_func(data[2]), ...)`

Key parameters:
- `work`: A user-defined function that takes an integer and returns its square.
- `aggregate`: A user-defined associative function that takes two integers and returns their sum.
- `data`: An array of integers from 1 to N (10,000 in this case).
- `chunksize`: The number of elements each task will process (500 in this case). If chunksize is not specified, 
               the AMT runtime system will automatically determine the optimal chunk size based on the number of available workers.

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3 map_reduce.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3,0 map_reduce.jl
"""

using torcjulia

work(x::Int)::Int = x^2

aggregate(x::Int, y::Int)::Int = x + y

function main()
    N = 10_000; chunksize = 500
    
    data = 1:N

    println("[Map] Each task computes the square of $chunksize elements.")
    println("[Reduce] Each task aggregates $chunksize elements.")
    println("Expected number of tasks = $(2 * ceil(Int, N / chunksize)).")

    result = torcjulia.mapReduce(work, aggregate, data; chunksize = chunksize)

    @assert result == Base.reduce((a, b) -> aggregate(a, b), [work(x) for x in data])
end

torcjulia.init(main)