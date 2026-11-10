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
Experiment: Parallel Mandelbrot Set Computation using torcjulia (Workload 4)

Description: Parallel computation of the Mandelbrot set using torcjulia. The generated image has a resolution of 5192×5192 pixels,
             with a maximum of 200,000 iterations per pixel, and each task processes a block of 64×64 pixels, except at the image 
             boundaries, where smaller blocks are formed.
"""

using torcjulia

include(joinpath(@__DIR__, "src", "mandelbrot.jl"))
import .Mandelbrot

using Printf

const WIDTH = 5192 # Width of the generated image
const HEIGHT = 5192 # Height of the generated image
const MAX_ITER = 200_000 # Maximum number of iterations per pixel
const BLOCK_SIZE = 64 # Each task processes a BLOCK_SIZE×BLOCK_SIZE pixel block

const WARMUP_WIDTH = 1024 # Width of the warmup image
const WARMUP_HEIGHT = 1024 # Height of the warmup image
const WARMUP_MAX_ITER = 1000

const SAVE_PATH = joinpath(@__DIR__, "..", "data", "mandelbrot_torcjulia.png") # Path to save the generated image

function main()
    # Warmup run to avoid compilation overhead in the timing of the Mandelbrot set computation
    tasks = [torcjulia.submit(Mandelbrot.block_mandelbrot, x, min(x + BLOCK_SIZE, WARMUP_WIDTH), y, min(y + BLOCK_SIZE, WARMUP_HEIGHT),
                              WARMUP_WIDTH, WARMUP_HEIGHT, WARMUP_MAX_ITER) for y in 0:BLOCK_SIZE:WARMUP_HEIGHT-1 for x in 0:BLOCK_SIZE:WARMUP_WIDTH-1]

    torcjulia.wait(tasks)

    # Enable the work stealing mechanism
    torcjulia.enable_stealing()
    
    t0 = torcjulia.gettime()

    tasks = [torcjulia.submit(Mandelbrot.block_mandelbrot, x, min(x + BLOCK_SIZE, WIDTH), y, min(y + BLOCK_SIZE, HEIGHT),
                              WIDTH, HEIGHT, MAX_ITER) for y in 0:BLOCK_SIZE:HEIGHT-1 for x in 0:BLOCK_SIZE:WIDTH-1]

    torcjulia.wait(tasks)

    elapsed = torcjulia.gettime() - t0

    @printf("Time taken to compute the Mandelbrot set: %.4f seconds\n", elapsed)

    blocks = [torcjulia.result(t) for t in tasks]
    Mandelbrot.save_image(blocks, HEIGHT, WIDTH, BLOCK_SIZE, MAX_ITER, SAVE_PATH)    
end

torcjulia.init(main)