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
Experiments: Parallel Monte Carlo Simulation for European Call Option Pricing using torcjulia (Workload 2)

Description: A parallel Monte Carlo simulation to price a European call option. Stock prices are simulated using Geometric Brownian Motion (GBM) 
             and the option price is estimated as the discounted average of the simulated payoffs.

             In total, 10^11 simulations are performed, evenly distributed across 1000 tasks.
"""

using torcjulia

include(joinpath(@__DIR__, "src", "monte_carlo.jl"))
import .MonteCarlo

using Printf

const S0 = 100.0 # Initial stock price
const K = 105.0 # Strike price
const T = 1.0  # Time to maturity (in years)
const r = 0.05 # Risk-free interest rate (5% annually)
const SIGMA = 0.2 # Volatility (20% annually)
const NTRIALS = 100_000_000_000 # Number of Monte Carlo simulations

const SEED = 69
const BATCHSIZE = 64

const NTASKS = 1000
const WARMUP_NTRIALS = 1_000_000

function main()
    # Warmup run to avoid compilation overhead in the timing of the Monte Carlo simulation
    _ = MonteCarlo.mc_call_price(S0, K, T, r, SIGMA, WARMUP_NTRIALS, NTASKS, BATCHSIZE, SEED)

    t0 = torcjulia.gettime()

    option_price, std_error = MonteCarlo.mc_call_price(S0, K, T, r, SIGMA, NTRIALS, NTASKS, BATCHSIZE, SEED)

    elapsed = torcjulia.gettime() - t0

    @printf("Estimated European call option price: %.6f ± %.8f\n", option_price, std_error)
    @printf("Time taken to compute the option price: %.4f seconds\n", elapsed)
end

torcjulia.init(main)