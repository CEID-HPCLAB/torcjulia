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
Example: Asynchronous Parallel Map Algorithmic Skeleton

Description: Demonstrates the usage of the asynchronous parallel `map_async` algorithmic skeleton in torcjulia.
             `map_async` corresponds to the asynchronous variant of the `map` algorithmic skeleton.

Notes: Unlike `map` which blocks execution until all tasks complete, `map_async` returns immediately with an AsyncResult handle, 
       allowing the caller to continue execution while tasks are processed in the background. The AsyncResult can be queried for 
       completion status and results.

       AsyncResult provides three helper functions:
       - `ready(ar)`: check if all spawned tasks have completed (non-blocking).
       - `wait(ar)`: block until all spawned tasks complete.
       - `get(ar)`: block until the tasks complete and collect results into an array.

                            map_async(f, data)
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

      `map_async` supports also an optional callback that executes asynchronously once all related tasks have completed, 
      receiving the collected results as input.

In this example, we define five functions: `fetch_data`, `work1`, `work2`, `work_batch`, and `aggregate`. The `fetch_data` function simulates retrieving data 
from a remote source, which is then processed by `work2`. The results produced by `work2` are subsequently combined using `aggregate`. Meanwhile, 
`work1` performs an independent computation on a vector of integers. The `work_batch` function processes a batch of integers 
and returns twice their sum. Finally, the total score is calculated by summing the results returned by `aggregate`, `work_batch`, and `work1`. 

The execution flow is as follows:

    data = 1:N          vec = [...]          arr = [[...], [...], ...]
         |                  |                         |
         v                  v                         v
    fetch_data            work1                  work_batch
         |                  |                         |
         v                  v                         v
     fetch_fut          work1_fut                 batch_fut
    (AsyncResult)      (AsyncResult)             (AsyncResult)
         |                  |                         |
         v                  v                         v
    ready() / get()    ready() / get()           ready() / get()
         |                  |                         |
         v                  v                         v
    fetched_data      work1_results              batch_results
         |                  |                         |
         v                  |                         |
     work2_fut              |                         |
         |                  |                         |
         v                  |                         |
    ready() / get()         |                         |
         |                  |                         |
         v                  |                         |
    work2_results           |                         |
         |                  |                         |
         v                  |                         |
      agg_fut               |                         |
    (aggregate)             |                         |
         |                  |                         |
         v                  |                         |
    ready() / get()         |                         |
         |                  |                         |
         v                  |                         |
     agg_results            |                         |
         |                  |                         |
         +------------------+-------------------------+
                            |
                            v
    total_score = agg_results["total"] + Base.sum(batch_results) + Base.sum(work1_results)

Julia < v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 5 map_async.jl

Julia >= v1.12:
Run: mpiexecjl -n 2 julia --project=/path/to/torcjulia/project --threads 5,0 map_async.jl
"""

using torcjulia

function fetch_data(id::Int; verbose::Bool = true)::Dict{String, Int}
    sleep(1)
    
    if verbose
        println("[Fetch_data] -> ID $id.")
    end

    Dict("id" => id, "value" => id * 3)
end

function work1(n::Int; verbose::Bool = true)::Int
    sleep(0.5)
    
    result = Base.sum(i^2 for i in 1:n)
    
    if verbose
        println("[Work1] -> N: $n - Result: $result.")
    end

    result
end

function work2(data::Dict{String, Int}; verbose::Bool = true)::Dict{String, Int}
    sleep(rand(0.3:0.2:1.3))
    
    result = data["value"] * 2 + 10
    
    if verbose
        println("[Work2] -> ID $(data["id"]), Value: $(data["value"]) - Result: $result.")
    end

    Dict("id" => data["id"], "processed_val" => result)
end

function aggregate(results::Vector{Dict{String, Int}}; verbose::Bool = true)::Dict{String, Float64}
    sleep(1.0)
    
    total = Base.sum(r["processed_val"] for r in results)
    avg = total / length(results)
    
    if verbose
        println("[Aggregate] -> Total: $total, Average: $avg.")
    end

    Dict("total" => total, "average" => avg)
end

function work_batch(batch::Vector{Int}; verbose::Bool = true)::Int
    sleep(0.5)
    
    result = Base.sum(batch) * 2
    
    if verbose
        println("[Work_batch] -> Batch: $batch - Result: $result.")
    end

    result
end

function main()
    N = 4
    
    data = 1:N
    
    vec = [15, 30, 10]
    arr = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]  

    # Set verbose to false to reduce output clutter
    verbose = true
    
    torcjulia.enable_stealing()

    println("[Work Stealing] -> Enabled.")
    
    fetch_fut = torcjulia.map_async(fetch_data, data; verbose = verbose,
        callback = r -> println("[Callback] -> Fetch complete: $(length(r)) items."))
    
    work1_fut = torcjulia.map_async(work1, vec; verbose = verbose,
        callback = r -> println("[Callback] -> Work1 complete - Result: $r."))
    
    !torcjulia.ready(fetch_fut) ? println("Waiting for fetch results...") : nothing
    
    fetched_data = torcjulia.get(fetch_fut)
    
    work2_fut = torcjulia.map_async(work2, fetched_data; verbose = verbose, chunksize = 2,
        callback = r -> println("[Callback] -> Work2 complete: $(length(r)) items."))
    
    !torcjulia.ready(work1_fut) ? println("Waiting for work1 results...") : nothing
    
    work1_results = torcjulia.get(work1_fut)

    !torcjulia.ready(work2_fut) ? println("Waiting for work2 results...") : nothing
    
    work2_results = torcjulia.get(work2_fut)
    
    agg_fut = torcjulia.map_async(aggregate, [work2_results]; verbose = verbose,
        callback = r -> println("[Callback] -> Aggregation complete - Result: $(r[1]["average"]) (Average), $(r[1]["total"]) (Total)."))
    
    batch_fut = torcjulia.map_async(work_batch, arr; verbose = verbose,
        callback = r -> println("[Callback] -> Batch processing complete - Result: $r."))
    
    !torcjulia.ready(agg_fut) ? println("Waiting for aggregation results...") : nothing
    
    !torcjulia.ready(batch_fut) ? println("Waiting for batch processing results...") : nothing
    
    agg_results = torcjulia.get(agg_fut); batch_results = torcjulia.get(batch_fut)
    
    total_score = agg_results[1]["total"] + Base.sum(batch_results) + Base.sum(work1_results)

    expected_score = aggregate(
                        Base.map(x -> work2(x; verbose = false),
                            Base.map(y -> fetch_data(y; verbose = false), data));
                        verbose = false
                    )["total"] +
                    Base.sum(Base.map(batch -> work_batch(batch; verbose = false), arr)) +
                    Base.sum(Base.map(n -> work1(n; verbose = false), vec))

    @assert total_score == expected_score
end

torcjulia.init(main)