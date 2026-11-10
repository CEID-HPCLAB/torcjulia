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
Example: Parallel StarMap Algorithmic Skeleton

Description: Demonstrates the usage of the parallel `starmap` algorithmic skeleton in torcjulia.

Notes: The parallel `starmap` skeleton applies a user-defined function to each element of the input iterable, 
       unpacking the elements of each tuple as separate arguments to the function. In this example, we define a 
       simple function `work` that takes four integer arguments and computes a simple arithmetic expression.

Key difference from `map`:
- `map(f, [(1,2), (3,4)])` → `[f((1,2)), f((3,4))]` (passes tuple as single arg)
- `starmap(f, [(1,2), (3,4)])` → `[f(1,2), f(3,4)]` (unpacks tuple elements as separate args)

Illustrative example: `starmap(work, [(1, 2), (3, 4)])` → `[work(1, 2), work(3, 4)]`

Key parameters:
- `work`: A user-defined function that takes four integer arguments and returns an integer result.
- `data`: An array of tuples, each containing two integers. The other two arguments are provided as additional parameters to the `starmap` function.
- `chunksize`: The number of elements each task will process (1,000 in this case). If chunksize is not specified, 
               the AMT runtime system will automatically determine the optimal chunk size based on the number of available workers.

Key points:
- The `starmap` function is a slightly modified parallel version of the `map` function, where each element of the 
  input iterable is unpacked and passed as separate arguments to the function.
- The `args` parameter allows you to pass additional arguments to the function being applied.

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3 starmap.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3,0 starmap.jl
"""

using torcjulia

work(x::Int, y::Int, a::Int, b::Int)::Int = (x + y + a) * b

function main()
    N = 10_000; chunksize = 1_000
    
    data = [(i, i * 2) for i in 1:N]
    
    a = 4; b = 2

    println("Each task computes the work function for $chunksize elements.")
    println("Expected number of tasks = $(ceil(Int, N / chunksize)).")
    
    results = torcjulia.starmap(work, data; chunksize = chunksize, args = (a, b))

    @assert results == [(x + y + a) * b for (x, y) in data]
end

torcjulia.init(main)