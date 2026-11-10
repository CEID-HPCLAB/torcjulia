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

module Mandelbrot

export save_image

using Colors, ColorSchemes, Images, FileIO

function _shading(norm_data::Matrix{Float64})::Matrix{Float64}
    gx, gy = _gradient(norm_data)
    
    intensity = sqrt.(gx.^2 .+ gy.^2); clamp.(norm_data .- 0.5*intensity, 0, 1)
end

function _gradient(img::Matrix{Float64})::Tuple{Matrix{Float64}, Matrix{Float64}}
    h, w = size(img)
    gx = zeros(Float64, h, w)
    gy = zeros(Float64, h, w)

    @views for i in 1:h
        gx[i,2:w-1] .= (img[i,3:w] .- img[i,1:w-2]) ./ 2
        gx[i,1] = img[i,2] - img[i,1]
        gx[i,w] = img[i,w] - img[i,w-1]
    end

    @views for j in 1:w
        gy[2:h-1,j] .= (img[3:h,j] .- img[1:h-2,j]) ./ 2
        gy[1,j] = img[2,j] - img[1,j]
        gy[h,j] = img[h,j] - img[h-1,j]
    end

    gx, gy
end

function save_image(blocks::Vector{Matrix{Float64}}, height::Int, width::Int, block_size::Int, max_iter::Int, path::String)
    image_data = Matrix{Float64}(undef, height, width)
    
    nx = cld(width, block_size)

    for (k, block) in enumerate(blocks)
        x = ((k - 1) % nx) * block_size
        y = ((k - 1) ÷ nx) * block_size

        image_data[y+1:y+size(block, 1), x+1:x+size(block, 2)] = block
    end

    norm_data = image_data ./ max_iter
    shaded_data = _shading(norm_data)

    cmap = ColorSchemes.magma
    colored_img = colorview(RGB, Base.map(x -> get(cmap, x), shaded_data))

    save(path, colored_img); println("Mandelbrot image saved to: $path.")
end

end