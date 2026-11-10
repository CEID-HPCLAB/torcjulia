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

export block_mandelbrot, save_image

using Colors, ColorSchemes, Images, FileIO
using LinearAlgebra

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