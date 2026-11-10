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
Example: Parallel Reduce Algorithmic Skeleton

Description: Demonstrates the usage of the parallel `reduce` algorithmic skeleton in torcjulia.

Notes: The parallel `reduce` skeleton applies an associative reduction operation across matrix or vector elements. 
       For matrices, it supports different reduction directions (:row or :column), and two modes: "1d" (reduces 
       to a single value per row/column) and "2d" (progressive reduction: matrix → vector → scalar). In this example, 
       we define a simple function `work` that computes the sum of three integers.

Key parameters:
- `work`: A user-defined function that takes three integer arguments and returns an integer result.
- `data`: An array (vector or matrix) of integers.
- `chunksize`: The number of elements each task will process. If chunksize is not specified, 
               the AMT runtime system will automatically determine the optimal chunk size based on the number of available workers.
- `direction`: For matrices, specifies the reduction direction (:row or :column). Ignored for vectors.
               :column → reduces along columns (aggregates rows): (mxn) → (1xn) or scalar
               :row → reduces along rows (aggregates columns): (mxn) → (mx1) or scalar
- `mode`: For matrices, specifies the reduction mode ("1d" or "2d"). "1d" reduces to a single value per row/column, 
          while "2d" performs a progressive reduction to a scalar.

Illustrative examples: 
Vector: `reduce(+, [1, 2, 3, 4, 5])` → `15`
Matrix (1d mode): `reduce(+, A, direction = :column)` → `[col1_sum, col2_sum, col3_sum, ...]`
Matrix (1d mode): `reduce(+, A, direction = :row)` → `[row1_sum, row2_sum, row3_sum, ...]`
Matrix (2d mode): `reduce(+, A)` → matrix → vector → scalar

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3 reduce.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3,0 reduce.jl
"""

using torcjulia

using Random
   
work(a::Int, b::Int, c::Int)::Int = a + b + c

function main()
    Random.seed!(1234)
    
    vec = rand(1:100, 50_000)
    mat = rand(1:15, 1_000, 100)

    println("Vector size: $(length(vec)).")
    println("Matrix size: $(size(mat)).")
    
    result_vec = torcjulia.reduce(
        work,
        vec;
        args = (10,),
        chunksize = 1_000
    )

    @assert result_vec == Base.reduce((a, b) -> work(a, b, 10), vec)
    
    result_1d_col = torcjulia.reduce(
        work,
        mat;
        args = (5,),
        chunksize = 500,
        direction = :column
    )
    
    @assert result_1d_col == [Base.reduce((a, b) -> work(a, b, 5), mat[:, i]) for i in 1:size(mat, 2)]
    
    result_1d_row = torcjulia.reduce(
        work,
        mat;
        args = (-2,),
        chunksize = 20,
        direction = :row
    )
    
    @assert result_1d_row == [Base.reduce((a, b) -> work(a, b, -2), mat[i, :]) for i in 1:size(mat, 1)]
    
    result_2d = torcjulia.reduce(
        work,
        mat;
        args = (8,),
        chunksize = 500,
        mode = "2d"
    )

    @assert result_2d == Base.reduce((a, b) -> work(a, b, 8), [Base.reduce((a, b) -> work(a, b, 8), mat[:, i]) for i in 1:size(mat, 2)])
end

torcjulia.init(main)