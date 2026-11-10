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

function mc_call_price_chunk(args::Tuple{Float64, Float64, Float64, Float64, Int, Int, Int})::Tuple{Float64, Float64}
    S0, diffusion, K, r, ntrials, batch_size, seed = args
    
    rng = Xoshiro(seed)

    Z_buf = Vector{Float64}(undef, batch_size)
    _base, _rem = divrem(ntrials, batch_size)

    local_sum = 0.0; local_sum_sq = 0.0

    for _ in 1:_base
        randn!(rng, Z_buf)
        @fastmath @inbounds for i in 1:batch_size
            ST = S0 * exp(diffusion * Z_buf[i])
            payoff = max(ST - K, 0.0)
            local_sum += payoff; local_sum_sq += payoff^2
        end
    end

    if _rem > 0
        randn!(rng, view(Z_buf, 1:_rem))
        @fastmath @inbounds for i in 1:_rem
            ST = S0 * exp(diffusion * Z_buf[i])
            payoff = max(ST - K, 0.0)
            local_sum += payoff; local_sum_sq += payoff^2
        end

    end

    local_sum, local_sum_sq
end