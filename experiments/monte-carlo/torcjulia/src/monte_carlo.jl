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

module MonteCarlo

using torcjulia

include("gbm.jl")

using Statistics, Random

function mc_call_price(S0::Float64, K::Float64, T::Float64, r::Float64, sigma::Float64, 
                       ntrials::Int, ntasks::Int, batch_size::Int, seed::Int)::Tuple{Float64, Float64}
    drift = (r - 0.5 * sigma^2) * T
    diffusion = sigma * sqrt(T)

    S0_drifted = S0 * exp(drift)
    
    base, rem = divrem(ntrials, ntasks)
    chunks = fill(base, ntasks); chunks[end] += rem

    args = [(S0_drifted, diffusion, K, r, chunks[i], batch_size, seed + i) for i in 1:ntasks]
    tasks = torcjulia.map(mc_call_price_chunk, args; chunksize = 1)

    mean_payoff = sum(first(t) for t in tasks) / ntrials
    total_payoffs_sq = sum(last(t) for t in tasks)

    disc_factor = exp(-r * T)
    option_price = disc_factor * mean_payoff
    
    var = (total_payoffs_sq / ntrials) - mean_payoff^2
    std_error = disc_factor * sqrt(max(var, 0.0) / ntrials)

    option_price, std_error
end

end