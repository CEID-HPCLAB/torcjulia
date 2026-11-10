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
Example: Parallel Scan Algorithmic Skeleton

Description: Demonstrates the usage of the parallel `scan` algorithmic skeleton in torcjulia.

Notes: The parallel `scan` skeleton computes prefix results over an input iterable using a user-defined associative function. 
       It supports both inclusive and exclusive scans. In this example, the function work adds two integers and optionally subtracts 
       a user-defined parameter k from their sum.

Illustrative example: `scan(work, [1, 2, 3, 4])` → `[1, 3, 6, 10]` (inclusive scan)
                      `scan(work, [1, 2, 3, 4]; mode = "exclusive")` → `[0, 1, 3, 6]` (exclusive scan)

Key parameters:
- `work`: A user-defined associative function that takes two integer arguments and returns their sum,
          with an optional parameter `k` that can be used to adjust the result.
- `data`: An array of integers from 1 to N (100 in this case).
- `chunksize`: The number of elements each task will process (10 in this case). If chunksize is not specified, 
               the AMT runtime system will automatically determine the optimal chunk size based on the number of available workers.
- `mode`: A string that specifies whether to perform an "inclusive" or "exclusive" scan. The default is "inclusive".
- `identity`: The identity element for the exclusive scan. The default is 0.

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3 scan.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 3,0 scan.jl
"""

using torcjulia

work(x::Int, y::Int; k::Int = 0)::Int = x + y - k

function main()
    N = 100; chunksize = 10
    
    data = collect(Int64, 1:N)
    
    results_excl = torcjulia.scan(work, data; chunksize = chunksize, mode = "exclusive")
    
    @assert results_excl == [0; cumsum(data[1:end-1])]

    results_incl = torcjulia.scan(work, data; chunksize = chunksize, mode = "inclusive")

    @assert results_incl == cumsum(data)
end

torcjulia.init(main)