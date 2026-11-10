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
Experiment: Parallel Non-Linear Optimization using the Hooke-Jeeves Algorithm with torcjulia (Workload 1)

Description: Finding the minimum of the Rosenbrock ("banana") function using the Hooke-Jeeves algorithm [1] with torcjulia. 
             The Julia implementation of the Hooke-Jeeves algorithm is adapted from the Algol pseudocode provided in [2]. 
             
             The optimization is performed over 32 variables, with the search space defined in the range [-5, 5] for each variable.
             The workload consists of 500 tasks, each executing 400 runs with a maximum of 5000 iterations per run.

References:
[1] Robert Hooke and T.A. Jeeves. "Direct Search" Solution of Numerical and Statistical Problems. 
    Journal of the ACM 8, 2 (1961), 212–229. doi:10.1145/321062.321069
[2] A.F. Kaupe, Algorithm 178: Direct Search. Communications of the ACM 6, 6 (1963), 313–314. doi:10.1145/366604.366617
"""

using torcjulia

include(joinpath(@__DIR__, "src", "hooke_jeeves.jl"))
import .HookeJeeves

using Printf

const RHO_BEGIN = 0.9 # Stepsize geometric shrink
const EPSMIN = 1e-6 # Ending value of stepsize
const MAX_ITER = 5000 # Maximum number of iterations

const LOW = -5.0
const HIGH = 5.0

const SEED = 69 

const NVARS = 32 # Number of variables 
const NTRIALS = 200_000 # Number of runs

const NTASKS = 500 # Number of tasks to split the optimization into
const WARMUP_NTRIALS = 10_000

const WORK_STEALING = parse(Bool, Base.get(ENV, "WORK_STEALING", "true"))

@inline function rosenbrock(x::Vector{Float64})::Float64
    result = 0.0
    
    @inbounds for i in 1:(length(x)-1)
        diff = x[i+1] - x[i]^2
        result += 100.0 * diff^2 + (1.0 - x[i])^2
    end
    
    result
end

function main()    
    warmup_base, warmup_rem = divrem(WARMUP_NTRIALS, NTASKS)
    warmup_chunks = fill(warmup_base, NTASKS); warmup_chunks[end] += warmup_rem

    warmup_args = [(rosenbrock, i, LOW, HIGH, RHO_BEGIN, MAX_ITER, EPSMIN, NVARS, warmup_chunks[i], SEED + i) for i in 1:NTASKS]
    
    # Warmup run to avoid compilation overhead in the timing of the optimization
    _ = torcjulia.starmap(HookeJeeves.optimize, warmup_args; chunksize = 1)

    base, rem = divrem(NTRIALS, NTASKS)
    chunks = fill(base, NTASKS); chunks[end] += rem

    best_fx = Inf; best_x = zeros(NVARS)
    best_iters = 0; best_trial = 0; total_evals = 0

    args = [(rosenbrock, i, LOW, HIGH, RHO_BEGIN, MAX_ITER, EPSMIN, NVARS, chunks[i], SEED + i) for i in 1:NTASKS]

    if WORK_STEALING
        torcjulia.enable_stealing() # Enable the work stealing mechanism
    end
    
    t0 = torcjulia.gettime()
    
    partial_results = torcjulia.starmap(HookeJeeves.optimize, args; chunksize = 1)

    for res in partial_results
        f, x, iters_trial, trial, evals = res
        total_evals += evals
        
        if f < best_fx
            best_fx = f; best_x = x
            best_iters = iters_trial; best_trial = trial
        end
    end

    elapsed = torcjulia.gettime() - t0

    @printf("Total trials: %d, Total evaluations: %d\n", NTRIALS, total_evals)
    @printf("Best f(x): %.6e\n", best_fx)
    @printf("Best x[1:5]: %s\n", string(best_x[1:5]))
    @printf("Best result at trial %d used %d iterations\n", best_trial, best_iters)
    
    @printf("Time taken to complete the optimization: %.4f seconds\n", elapsed)
end

torcjulia.init(main)