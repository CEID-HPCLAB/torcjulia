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
Example: Asynchronous Parallel Processing in torcjulia using `map_async` and `starmap_async`.

Description: This example demonstrates the usage of the asynchronous parallel `map_async` and `starmap_async` algorithmic skeletons in torcjulia. 
             These skeletons allow for non-blocking execution of tasks, enabling concurrent processing of different task functions.

(For details on each skeleton, see `map_async.jl` and `starmap_async.jl`)

Execution flow:

    pairs = [(1,10), (2,20), ...]          numbers = [1, 2, ..., 10]
                 |                                      |
                 v                                      v
          starmap_async(work1)                   map_async(work2)
                 |                                      |
                 v                                      v
               fut1  <------ executed concurrently ----> fut2
                 |                                      |
                 v                                      v
              get(fut1)                              get(fut2)
                 |                                      |
                 v                                      v
             results1                                results2
                 |                                      |
                 '------------------.-------------------'
                                    |
                                    v
                 map_async(aggregate, [results1, results2])
                                    |
                                    v
                                 results3
                                    |
                                    v
       final_score = results3[1]["total"] + results3[2]["total"]

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 5 async_processing.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 5,0 async_processing.jl
"""

using torcjulia

function work1(a::Int, b::Int; verbose::Bool = true)::Int
    sleep(0.7 + rand())
    
    result = a * 2 + b * 2
    
    if verbose
        println("[Work1] -> (a: $a, b: $b) - Result: $result.")
    end
    
    result
end

function work2(x::Int; verbose::Bool = true)::Int
    sleep(1 + rand())
    
    result = x^2
    
    if verbose
        println("[Work2] -> (x: $x) - Result: $result.")
    end
    
    result
end

function aggregate(values::Vector{Int}; verbose::Bool = true)::Dict{String, Float64}
    sleep(0.6)
    
    total = Base.sum(values)
    avg = total / length(values)
    
    if verbose
        println("[Aggregate] -> Total: $total, Average: $avg.")
    end
    
    Dict("total" => total, "average" => avg)
end

function main()
    N = 16
    
    pairs = [(i, i * N) for i in 1:N]
    numbers = collect(1:N)

    # Set verbose to false to reduce output clutter
    verbose = true

    fut1 = torcjulia.starmap_async(work1, pairs; verbose = verbose,
        callback = r -> println("[Callback] -> Starmap (Work1) complete: $(length(r)) items."))

    fut2 = torcjulia.map_async(work2, numbers; verbose = verbose, chunksize = 4,
        callback = r -> println("[Callback] -> Map (Work2) complete: $(length(r)) items."))

    results1 = torcjulia.get(fut1)
    
    results2  = torcjulia.get(fut2)

    fut3 = torcjulia.map_async(aggregate, [results1, results2]; verbose = verbose,
        callback = r -> println("[Callback] -> Aggregation complete: $(length(r)) items."))

    results3 = torcjulia.get(fut3)

    total_score = results3[1]["total"] + results3[2]["total"]

    expected_score = Base.sum(
        aggregate(vals; verbose = false)["total"]
        for vals in (
            [work1(a, b; verbose = false) for (a, b) in [(i, i*N) for i in 1:N]],
            [work2(x; verbose = false) for x in 1:N]
        )
    )
    
    @assert total_score == expected_score
end

torcjulia.init(main)