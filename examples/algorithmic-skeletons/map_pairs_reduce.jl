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
Example: Parallel MapPairsReduce Algorithmic Skeleton

Description: Demonstrates the usage of the parallel `mapPairsReduce` algorithmic skeleton in torcjulia.

Notes: The parallel `mapPairsReduce` skeleton encapsulates the functionality of both `mapPairs` and `reduce` skeletons. It applies 
       a user-defined function to every (x, y) pair from the Cartesian product of two input arrays, producing a result matrix, and 
       then reduces the results using an associative operation. In this example, we define a simple function `work` that computes 
       the sum of two integers and a reduce function `aggregate` that computes the maximum of two floating-point numbers.
       Reduction can be performed in either "1d" or "2d" mode, with the option to specify the direction of reduction for "1d" mode (either :row or :column).
       A detailed description regarding the reduction modes and directions can be found in the `reduce.jl` example.

Illustrative example: `mapPairsReduce(work, aggregate, [1, 2], [10, 20])` → `aggregate(work(1, 10), work(1, 20), work(2, 10), work(2, 20))`

Key parameters:
- `work`: A user-defined function that takes two integer arguments and returns their sum.
- `aggregate`: A user-defined associative function that takes two floating-point numbers and returns their maximum.
- `a`, `b`: Two input arrays of integers, each containing N elements (2,000 and 300 in this case).
- `chunksize`: The number of pairs each task will process (5,000 in this case). If chunksize is not specified, 
               the AMT runtime system will automatically determine the optimal chunk size based on the number of available workers.
- `mode`: The mode of reduction, either "1d" or "2d".
- `direction`: The direction of reduction for "1d" mode, either :row or :column.

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 5 map_pairs_reduce.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 5,0 map_pairs_reduce.jl
"""

using torcjulia

work(a::Int, b::Int)::Float64 = a + b

aggregate(x::Float64, y::Float64)::Float64 = max(x, y)

function main()
    N_a = 2_000; N_b = 300
    
    a = 1:N_a; b = N_b:-1:1

    result_1d_row = torcjulia.mapPairsReduce(work, aggregate, a, b; mode = "1d", direction = :row)
    
    @assert result_1d_row == [Base.reduce((x, y) -> aggregate(x, y), [work(x, y) for y in b]) for x in a]

    result_1d_col = torcjulia.mapPairsReduce(work, aggregate, a, b; mode = "1d", direction = :column)
    
    @assert result_1d_col == [Base.reduce((x, y) -> aggregate(x, y), [work(x, y) for x in a]) for y in b]

    result_2d = torcjulia.mapPairsReduce(work, aggregate, a, b; mode = "2d")
    
    @assert result_2d == Base.reduce((x, y) -> aggregate(x, y), [work(x, y) for x in a, y in b])
end

torcjulia.init(main)