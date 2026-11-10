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

module HookeJeeves

export optimize

include("optimization.jl")

using Random, LinearAlgebra, Distributions

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

end