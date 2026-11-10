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
Example: Asynchronous Parallel Starmap Algorithmic Skeleton

Description: Demonstrates the usage of the asynchronous parallel `starmap_async` algorithmic skeleton in torcjulia.
             `starmap_async` corresponds to the asynchronous variant of the `starmap` algorithmic skeleton.

Notes: Unlike `starmap` which blocks execution until all tasks complete, `starmap_async` returns immediately with an AsyncResult handle, 
       allowing the caller to continue execution while tasks are processed in the background. The AsyncResult can be queried for 
       completion status and results.

       AsyncResult provides three helper functions:
       - `ready(ar)`: check if all spawned tasks have completed (non-blocking)
       - `wait(ar)`: block until all spawned tasks complete
       - `get(ar)`: block until the tasks complete and collect results into a flat vector

                            starmap_async(f, data)
                                    |
                                    v
                                AsyncResult
                                    |
                                    v
                            .---------------.
                            |               |
                            v               v
                        ready()        get() / wait()
                        (non-blocking)        (blocking)
                            |               |
                            v               v
                        true / false   results / nothing

      `starmap_async` supports also an optional callback that executes asynchronously once all related tasks have completed, 
      receiving the collected results as input.

Key difference from `map_async`:
- `map_async(f, [(a,b), (c,d)])` → `[f((a,b)), f((c,d))]` (passes tuple as single arg)
- `starmap_async(f, [(a,b), (c,d)])` → `[f(a,b), f(c,d)]` (unpacks tuple elements as separate args)

Julia < v1.12
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 5 starmap_async.jl

Julia >= v1.12
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 5,0 starmap_async.jl
"""

using torcjulia

function work(x::Int, y::Int; scale::Float64 = 1.0, verbose::Bool = true)::Float64
    sleep(rand(0.1:0.2:1.3))  
    
    result = (x^2 + y^2) * scale
    
    if verbose
        println("[Work] -> (x: $x, y: $y) - Result: $result.")
    end
    
    result
end

function aggregate(a::Float64, b::Float64; verbose::Bool = true)::Float64
    sleep(0.2)
    
    result = a + b
    
    if verbose
        println("[Aggregate] -> (a: $a, b: $b) - Result: $result.")
    end
    
    result
end

function main()
    data1  = [(1, 2), (3, 4), (5, 6), (7, 8)]
    data2 = [(10, 20), (30, 40), (50, 60)]
    
    # Set verbose to false to reduce output clutter
    verbose = true
    
    torcjulia.enable_stealing()

    println("[Work Stealing] -> Enabled.")

    fut1 = torcjulia.starmap_async(work, data1; verbose = verbose,
        callback = r -> println("[callback] -> Data batch 1 complete - Result: $r."))

    fut2 = torcjulia.starmap_async(work, data2; scale = 2.0, verbose = verbose,
        callback = r -> println("[Callback] -> Data batch 2 complete - Result: $r."))

    !torcjulia.ready(fut1) ? println("Waiting for data batch 1...") : nothing
    
    results1 = torcjulia.get(fut1)

    !torcjulia.ready(fut2) ? println("Waiting for data batch 2...") : nothing
    
    results2 = torcjulia.get(fut2)

    pairs = collect(zip(results1, results2))
    
    fut3 = torcjulia.starmap_async(aggregate, pairs; verbose = verbose,
        callback = r -> println("[Callback] -> Aggregation complete - Result: $r."))

    results3 = torcjulia.get(fut3); total_score = Base.sum(results3)

    expected_score = Base.sum(
                        aggregate(a, b; verbose = false) 
                        for (a, b) in zip(
                            (work(x, y; verbose = false) for (x, y) in data1),         
                            (work(x, y; scale = 2.0, verbose = false) for (x, y) in data2) 
                        )
                    )
    
    @assert total_score == expected_score
end

torcjulia.init(main)