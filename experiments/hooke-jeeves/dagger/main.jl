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
Experiment: Parallel Non-Linear Optimization using the Hooke-Jeeves Algorithm with Dagger (Workload 1)

Description: Finding the minimum of the Rosenbrock ("banana") function using the Hooke-Jeeves algorithm [1] with Dagger [2]. 
             The Julia implementation of the Hooke-Jeeves algorithm is adapted from the Algol pseudocode provided in [3]. 
             
             The optimization is performed over 32 variables, with the search space defined in the range [-5, 5] for each variable.
             The workload consists of 500 tasks, each executing 400 runs with a maximum of 5000 iterations per run.

Notes: To ensure a fair comparison between Dagger and torcjulia, all experiments were conducted using identical process counts 
       (1, 2, 4, 8, etc.) and a fixed configuration of three Julia threads per process for both runtime systems.

Warning: According to [4], the main thread (Thread ID: 1) of the master process (Process ID: 1), which runs Dagger's core scheduler, 
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
[1] Robert Hooke and T.A. Jeeves. "Direct Search" Solution of Numerical and Statistical Problems. 
    Journal of the ACM 8, 2 (1961), 212–229. doi:10.1145/321062.321069
[2] Rabab Alomairy, Felipe Tome, Julian Samaroo, and Alan Edelman, Dynamic Task Scheduling with Data Dependency Awareness Using Julia. 
    In Proceedings of the 2024 High Performance Extreme Computing Conference (HPEC), 1–7. doi:10.1109/HPEC62836.2024.10938467
[3] A.F. Kaupe, Algorithm 178: Direct Search. Communications of the ACM 6, 6 (1963), 313–314. doi:10.1145/366604.366617
[4] Dagger.jl, Scheduler Internals - Online Documentation [Accessed: Oct. 14, 2026]. https://juliaparallel.org/Dagger.jl/stable/scheduler-internals/
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

using Dagger
using Printf

@everywhere using Random, LinearAlgebra, Distributions

const NTRIALS = 200_000 # Number of runs

const NTASKS = 500 # number of tasks to split the optimization into
const WARMUP_NTRIALS = 10_000

# If true, the main thread of the master process participates in task execution; if false, it is reserved for 
# running exclusively Dagger's core scheduler
const USE_MAIN_THREAD = parse(Bool, Base.get(ENV, "USE_MAIN_THREAD", "false"))

@everywhere begin
    const RHO_BEGIN = 0.9 # Stepsize geometric shrink
    const EPSMIN = 1e-6 # Ending value of stepsize
    const MAX_ITER = 5000 # Maximum number of iterations

    const LOW = -5.0
    const HIGH = 5.0

    const SEED = 69 

    const NVARS = 32 # Number of variables 

    @inline function rosenbrock(x::Vector{Float64})::Float64
        result = 0.0
        
        @inbounds for i in 1:(length(x)-1)
            diff = x[i+1] - x[i]^2
            result += 100.0 * diff^2 + (1.0 - x[i])^2
        end
        
        result
    end

    function optimize(obj_func, id::Int, low::Float64, high::Float64, rho_begin::Float64, max_iter::Int, 
                      epsmin::Float64, nvars::Int, ntrials::Int, seed::Int)::Tuple{Float64, Vector{Float64}, Int, Int, Int}
        rng = Xoshiro(seed)
        
        best_fx = Inf; best_x = zeros(nvars); best_trial = 0
        iters = 0; total_evals = 0

        for trial in 1:ntrials
            x0 = rand(rng, Uniform(low, high), nvars)
            
            f, x, iters_trial, evals = run_trial(x0, rho_begin, max_iter, epsmin, obj_func)
            total_evals += evals
            
            if f < best_fx
                best_fx, best_x, iters = f, x, iters_trial
                best_trial = (id - 1) * ntrials + trial
            end
        end
        
        best_fx, best_x, iters, best_trial, total_evals
    end

    function explore(x::Vector{Float64}, delta::Vector{Float64}, fbefore::Float64, obj_fn)::Tuple{Float64, Int}
        fmin = fbefore; funevals = 0
        
        @inbounds for i in 1:length(x)
            orig = x[i]; x[i] = orig + delta[i]
            fplus = obj_fn(x); funevals += 1
            
            if fplus < fmin
                fmin = fplus
            
            else
                delta[i] = -delta[i]; x[i] = orig + delta[i]
                fminus = obj_fn(x); funevals += 1
                
                if fminus < fmin
                    fmin = fminus
                else
                    x[i] = orig
                end
            end
        end
        
        fmin, funevals
    end

    function run_trial(x0::Vector{Float64}, rho_begin::Float64, max_iter::Int, 
                  epsmin::Float64, obj_fn)::Tuple{Float64, Vector{Float64}, Int, Int}
        
        n = length(x0)
        
        xbefore = copy(x0); xnew = Vector{Float64}(undef, n); delta = Vector{Float64}(undef, n)  

        @inbounds for i in 1:n
            delta[i] = abs(xbefore[i] * rho_begin) > 0 ? abs(xbefore[i] * rho_begin) : rho_begin
        end
        
        steplength = rho_begin; fbefore = obj_fn(xbefore)
        
        iters = 0; funevals = 1

        while steplength > epsmin && iters < max_iter
            iters += 1; copyto!(xnew, xbefore)  
            
            fnew, evals = explore(xnew, delta, fbefore, obj_fn)
            
            funevals += evals

            discard = true
            
            while fnew < fbefore && discard
                @inbounds for i in 1:n
                    delta[i] = xnew[i] <= xbefore[i] ? -abs(delta[i]) : abs(delta[i])
                    tmp = xbefore[i]; xbefore[i] = xnew[i]; xnew[i] = 2 * xnew[i] - tmp
                end
                
                fbefore = fnew; fnew, evals = explore(xnew, delta, fbefore, obj_fn)
                
                funevals += evals
                
                discard = false
                
                @inbounds for i in 1:n
                    if abs(xnew[i] - xbefore[i]) > 0.5 * abs(delta[i])
                        (discard = true) && break
                    end
                end
            end

            if fnew >= fbefore
                steplength *= rho_begin
                
                @inbounds for i in 1:n
                    delta[i] *= rho_begin
                end
            end
        end

        f_min = obj_fn(xbefore); funevals += 1
        
        f_min, xbefore, iters, funevals
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

    warmup_base, warmup_rem = divrem(WARMUP_NTRIALS, NTASKS)
    warmup_chunks = fill(warmup_base, NTASKS); warmup_chunks[end] += warmup_rem

    warmup_args = [(rosenbrock, i, LOW, HIGH, RHO_BEGIN, MAX_ITER, EPSMIN, NVARS, warmup_chunks[i], SEED + i) for i in 1:NTASKS]
    
    # Warmup run to avoid compilation overhead in the timing of the optimization
    tasks = [Dagger.@spawn scope = sc optimize(a...) for a in warmup_args]
    _ = fetch.(tasks)

    base, rem = divrem(NTRIALS, NTASKS)
    chunks = fill(base, NTASKS); chunks[end] += rem

    best_fx = Inf; best_x = zeros(NVARS)
    best_iters = 0; best_trial = 0; total_evals = 0

    args = [(rosenbrock, i, LOW, HIGH, RHO_BEGIN, MAX_ITER, EPSMIN, NVARS, chunks[i], SEED + i) for i in 1:NTASKS]
    
    t0 = time_ns()
    
    partial_results = fetch.(Dagger.@spawn scope = sc optimize(a...) for a in args)

    for res in partial_results
        f, x, iters_trial, trial, evals = res
        total_evals += evals
        
        if f < best_fx
            best_fx = f; best_x = x
            best_iters = iters_trial; best_trial = trial
        end
    end

    elapsed = (time_ns() - t0) / 1e9

    @printf("Total trials: %d, Total evaluations: %d\n", NTRIALS, total_evals)
    @printf("Best f(x): %.6e\n", best_fx)
    @printf("Best x[1:5]: %s\n", string(best_x[1:5]))
    @printf("Best result at trial %d used %d iterations\n", best_trial, best_iters)
    
    @printf("Time taken to complete the optimization: %.4f seconds\n", elapsed)
end

main()