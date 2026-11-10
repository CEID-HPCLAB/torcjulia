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
Experiment: Parallel Mandelbrot Set Computation using Dagger (Workload 4)

Description: Parallel computation of the Mandelbrot set using Dagger. The generated image has a resolution of 5192×5192 pixels,
             with a maximum of 200,000 iterations per pixel, and each task processes a block of 64×64 pixels, except at the image 
             boundaries, where smaller blocks are formed.

Notes: To ensure a fair comparison between Dagger and torcjulia, all experiments were conducted using identical process counts 
       (1, 2, 4, 8, etc.) and a fixed configuration of three Julia threads per process for both runtime systems.

Warning: According to [1], the main thread (Thread ID: 1) of the master process (Process ID: 1), which runs Dagger's core scheduler, 
         also participates in task execution. We confirmed this behavior experimentally. However, under this configuration, 
         Dagger exhibited high variability in wall-clock execution times across benchmark runs and often poor performance. We attributed 
         this behavior to the overhead associated with the main thread handling both task scheduling and execution.

         To investigate this issue, we used Dagger's scope feature to restrict task execution to processes 2 through n (n: total number of processes) 
         and to threads 2 and 3 of the master process, leaving the main thread (Thread ID: 1) of the master process exclusively responsible 
         for running Dagger's core scheduler. This alternative configuration yielded a modest improvement in Dagger's overall 
         performance, providing a second basis for comparison with torcjulia.

         We therefore used the latter configuration for all experiments reported in this work. To support reproducibility and enable evaluation 
         of both configurations, we provide the Dagger code for both approaches. A global configuration variable, `USE_MAIN_THREAD`, 
         determines which approach is used. By default, this variable is set to false, reserving the main thread of the master process exclusively 
         for running Dagger's core scheduler. Setting it to true allows the main thread to participate in task execution.

References:
[1] Dagger.jl, Scheduler Internals - Online Documentation [Accessed: Oct. 14, 2026]. https://juliaparallel.org/Dagger.jl/stable/scheduler-internals/
"""

using Distributed

using Sockets, ElasticClusterManager

@everywhere using Base.Threads

const SLURM_JOB_SCHEDULING = parse(Bool, Base.get(ENV, "SLURM_JOB_SCHEDULING", "false"))

if SLURM_JOB_SCHEDULING
    process_id = parse(Int, ENV["SLURM_PROCID"])
    num_processes = parse(Int, ENV["SLURM_NTASKS"])

    cookie = ENV["DAGGER_COOKIE"]                    
    job_id = ENV["SLURM_JOB_ID"]

    master_addr_file = joinpath(Base.get(ENV, "SLURM_SUBMIT_DIR", pwd()), ".dagger_master_$job_id")

    if process_id != 0
        t0 = time_ns() * 1e-9

        while !isfile(master_addr_file)
            time_ns() * 1e-9 - t0 > 200 && error("[Process $process_id] Timeout waiting for master address file: $master_addr_file.")
            sleep(0.2)
        end

        host, port = split(strip(read(master_addr_file, String)), ":")
        ElasticClusterManager.elastic_worker(cookie, String(host), parse(Int, port))
    end

    manager = ElasticManager(addr = :auto, port = 0, cookie = cookie)
    host = string(manager.sockname[1]); port = Int(manager.sockname[2])

    # Avoid race conditions
    tmp = master_addr_file * ".tmp"; write(tmp, "$host:$port"); mv(tmp, master_addr_file; force = true) 

    t0 = time_ns() * 1e-9

    while nprocs() < num_processes
        time_ns() * 1e-9 - t0 > 200 && error("[Process $process_id] Timeout waiting for all processes to join the cluster.")
        sleep(0.2)
    end

    rm(master_addr_file; force = true)

    for p in sort(procs())
        info = remotecall_fetch(p) do
            line = only(filter(startswith("Cpus_allowed_list:"), readlines("/proc/self/status")))

            (
                host = gethostname(),
                rank = get(ENV, "SLURM_PROCID", "?"),
                pid = getpid(),
                cpus = strip(split(line, ':')[2])
            )
        end

        println("Process $p | Node = $(info.host) | Rank = $(info.rank) | PID = $(info.pid) | CPU Cores = $(info.cpus)")
    end

else
    n_workers = parse(Int, Base.get(ENV, "N_WORKERS", "2")) 
    n_threads = parse(Int, Base.get(ENV, "N_THREADS", "3"))

    thread_args = VERSION >= v"1.12" ? "--threads=$(n_threads),0" : "--threads=$(n_threads)"

    addprocs(n_workers; exeflags = thread_args)
end

include(joinpath(@__DIR__, "src", "mandelbrot.jl"))
import .Mandelbrot

using Dagger
using Printf

const SAVE_PATH = joinpath(@__DIR__, "..", "data", "mandelbrot_dagger.png") # Path to save the generated image

@everywhere using LinearAlgebra

const BLOCK_SIZE = 64 # Each task processes a BLOCK_SIZE×BLOCK_SIZE pixel block

# If true, the main thread of the master process participates in task execution; if false, it is reserved for 
# running exclusively Dagger's core scheduler
const USE_MAIN_THREAD = parse(Bool, Base.get(ENV, "USE_MAIN_THREAD", "false"))

@everywhere begin
    const WIDTH = 5192 # Width of the generated image
    const HEIGHT = 5192 # Height of the generated image
    const MAX_ITER = 200_000 # Maximum number of iterations per pixel

    const WARMUP_WIDTH = 1024 # Width of the warmup image
    const WARMUP_HEIGHT = 1024 # Height of the warmup image
    const WARMUP_MAX_ITER = 1000

    function block_mandelbrot(x_start::Int, x_end::Int, y_start::Int, y_end::Int, 
                          width::Int, height::Int, max_iter::Int = 5000)::Matrix{Float64}

        ncols = x_end - x_start
        nrows = y_end - y_start
        
        block = Matrix{Float64}(undef, nrows, ncols)

        dx = 3.0 / (width - 1); dy = 2.0 / (height - 1)

        @inbounds for i in 1:nrows
            yi = -1.0 + (y_start + i - 1) * dy

            for j in 1:ncols
                xj = -2.0 + (x_start + j - 1) * dx
                zr = 0.0; zi = 0.0
                
                cnt = 0

                while zr * zr + zi * zi <= 4.0 && cnt < max_iter
                    tmp = zr

                    zr = zr * zr - zi * zi + xj
                    zi = 2.0 * tmp * zi + yi

                    cnt += 1
                end


                if cnt < max_iter
                    mag2 = zr * zr + zi * zi
                    block[i, j] = cnt + 1 - log(0.5 * log(mag2)) / log(2.0)
                
                else
                    block[i, j] = cnt
                end
            
            end
        end

        block
    end
end

function main()
    if USE_MAIN_THREAD
        sc = Dagger.UnionScope([Dagger.scope(worker = pid) for pid in 1:nprocs()]...)
    else
        # Exclude the main thread (Thread ID: 1) of the master process (Process ID: 1) from task execution
        master_worker_threads = Dagger.scope(worker = 1, threads = 2:Threads.nthreads()) 
    
        remote_workers = Any[Dagger.scope(worker = pid) for pid in workers()]
    
        sc = Dagger.UnionScope(master_worker_threads, remote_workers...)
    end

    # Warmup run to avoid compilation overhead in the timing of the Mandelbrot set computation
    tasks = [Dagger.@spawn scope = sc block_mandelbrot(x, min(x + BLOCK_SIZE, WIDTH), y, min(y + BLOCK_SIZE, HEIGHT),
                              WIDTH, HEIGHT, WARMUP_MAX_ITER) for y in 0:BLOCK_SIZE:WARMUP_HEIGHT-1 for x in 0:BLOCK_SIZE:WARMUP_WIDTH-1]

    _ = fetch.(tasks)

    t0 = time_ns()

    tasks = [Dagger.@spawn scope = sc block_mandelbrot(x, min(x + BLOCK_SIZE, WIDTH), y, min(y + BLOCK_SIZE, HEIGHT),
                              WIDTH, HEIGHT, MAX_ITER) for y in 0:BLOCK_SIZE:HEIGHT-1 for x in 0:BLOCK_SIZE:WIDTH-1]

    blocks = fetch.(tasks)

    elapsed = (time_ns() - t0) / 1e9

    @printf("Time taken to compute the Mandelbrot set: %.4f seconds\n", elapsed)

    Mandelbrot.save_image(blocks, HEIGHT, WIDTH, BLOCK_SIZE, MAX_ITER, SAVE_PATH)    
end

main()